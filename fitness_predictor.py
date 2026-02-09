"""
适应度预测器 (重构版)
使用有效的代理模型预测遗传算法个体的适应度，减少昂贵的评估次数

核心改进:
- 将染色体索引转换为有意义的特征（one-hot模型编码 + 实际超参数值）
- 使用RandomForest代替GBR（更适合异构特征和小样本）
- 基于排序的预测策略（不需要精确预测绝对适应度，只需正确排序）
"""
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


class FitnessPredictor:
    """适应度代理模型：基于特征工程的染色体适应度预测器
    
    核心思路:
    - 原始染色体 [model_idx, param1_idx, param2_idx, ...] 中的索引对不同模型含义不同
    - 需要解码为有意义的特征: [model_onehot..., actual_param_values...]
    - 这样代理模型才能学到 "XGBoost + n_estimators=200 + depth=5" → 高适应度
    """
    
    def __init__(self, model_hyperparameters=None):
        """
        :param model_hyperparameters: GA的模型超参数字典
            例: {'XGBoost': {'n_estimators': [50,100,200], 'max_depth': [3,5,7]}, ...}
        """
        # RandomForest 比 GBR 更适合：
        # 1. 对特征缩放不敏感（我们有one-hot和连续值混合）
        # 2. 天然处理特征交互（模型类型 × 超参数值）
        # 3. 小样本下不容易过拟合
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,          # 适度深度，防止过拟合
            min_samples_leaf=3,   # 每叶至少3样本
            max_features='sqrt',  # 随机子集，增加多样性
            random_state=42,
            n_jobs=1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.history = {
            'chromosomes': [],
            'fitnesses': []
        }
        self.min_samples_to_train = 20    # 最少20个样本才训练（降低门槛以更快启用）
        self.retrain_interval = 30        # 每30个新样本重新训练
        self.prediction_confidence = 0.0  # 预测置信度（基于交叉验证R²）
        self._samples_since_train = 0     # 上次训练后新增的样本数
        
        # 特征工程所需的模型信息
        self.model_hyperparameters = model_hyperparameters or {}
        self.model_names = list(self.model_hyperparameters.keys()) if model_hyperparameters else []
        self.n_models = len(self.model_names)
        
        # 计算特征维度
        self.max_params = max(
            (len(p) for p in self.model_hyperparameters.values()), default=0
        )
        self.feature_dim = self.n_models + self.max_params  # one-hot + params
        
        # 缓存解码后的特征，避免重复计算
        self._feature_cache = {}
        
        logging.info(f"适应度预测器初始化: {self.n_models}个模型, "
                     f"最大{self.max_params}个超参数, 特征维度={self.feature_dim}")
    
    def update_model_hyperparameters(self, model_hyperparameters):
        """当范围调整器更新超参数范围后，同步更新特征工程元信息"""
        self.model_hyperparameters = model_hyperparameters
        new_names = list(model_hyperparameters.keys())
        if new_names != self.model_names:
            self.model_names = new_names
            self.n_models = len(self.model_names)
        new_max = max((len(p) for p in model_hyperparameters.values()), default=0)
        if new_max != self.max_params:
            self.max_params = new_max
            self.feature_dim = self.n_models + self.max_params
            # 清除缓存因为特征维度改变了
            self._feature_cache.clear()
    
    def _extract_features(self, chromosome):
        """将染色体解码为有意义的特征向量
        
        关键创新: 不使用原始索引，而是:
        1. One-hot 编码模型类型 → 让模型学到不同模型的基线表现
        2. 实际超参数值 → 让模型学到参数值与适应度的关系
        
        :param chromosome: 原始染色体 [model_idx, param1_idx, param2_idx, ...]
        :return: 特征向量 [model_onehot..., normalized_param_values...]
        """
        # 使用元组作为缓存键
        cache_key = tuple(chromosome)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        features = []
        
        # 1. 模型类型 one-hot 编码
        model_idx = min(chromosome[0], self.n_models - 1) if self.n_models > 0 else 0
        model_onehot = [0.0] * self.n_models
        if model_idx < self.n_models:
            model_onehot[model_idx] = 1.0
        features.extend(model_onehot)
        
        # 2. 实际超参数值（归一化）
        param_features = [0.0] * self.max_params
        
        if model_idx < self.n_models:
            model_name = self.model_names[model_idx]
            params = self.model_hyperparameters.get(model_name, {})
            
            for i, (param_name, values) in enumerate(params.items()):
                if i >= self.max_params:
                    break
                
                gene_pos = i + 1  # +1 因为第0位是模型索引
                if gene_pos < len(chromosome) and len(values) > 0:
                    value_idx = min(chromosome[gene_pos], len(values) - 1)
                    val = values[value_idx]
                    
                    # 将值转换为数值特征
                    param_features[i] = self._value_to_feature(val, values)
                else:
                    param_features[i] = 0.0
        
        features.extend(param_features)
        
        result = np.array(features, dtype=float)
        self._feature_cache[cache_key] = result
        return result
    
    @staticmethod
    def _value_to_feature(val, all_values):
        """将超参数值转换为归一化数值特征
        
        策略:
        - None → -1.0 (特殊标记)
        - 数值 → 在该参数范围内的归一化位置 [0, 1]
        - 字符串/元组 → 在值列表中的位置索引归一化
        """
        if val is None:
            return -1.0
        
        if isinstance(val, (int, float, np.integer, np.floating)):
            # 数值参数：归一化到 [0, 1] 范围
            numeric_vals = [v for v in all_values 
                          if v is not None and isinstance(v, (int, float, np.integer, np.floating))]
            if len(numeric_vals) >= 2:
                min_v = min(numeric_vals)
                max_v = max(numeric_vals)
                if max_v > min_v:
                    return (float(val) - min_v) / (max_v - min_v)
            return 0.5  # 单值或全相同
        
        # 非数值参数（字符串、元组等）：使用位置索引归一化
        try:
            idx = all_values.index(val)
            return idx / max(len(all_values) - 1, 1)
        except (ValueError, TypeError):
            return 0.5
    
    def add_history(self, chromosome, fitness):
        """添加历史数据"""
        self.history['chromosomes'].append(list(chromosome))
        self.history['fitnesses'].append(fitness)
        self._samples_since_train += 1
        
        # 清除此染色体的缓存（因为特征可能因范围调整而改变）
        cache_key = tuple(chromosome)
        if cache_key in self._feature_cache:
            del self._feature_cache[cache_key]
    
    def should_retrain(self):
        """判断是否应该重新训练"""
        n = len(self.history['fitnesses'])
        if n < self.min_samples_to_train:
            return False
        if not self.is_trained:
            return True
        return self._samples_since_train >= self.retrain_interval
        
    def train(self):
        """训练预测模型 - 使用特征工程后的数据"""
        n = len(self.history['chromosomes'])
        if n < self.min_samples_to_train:
            return False
        
        if self.n_models == 0:
            logging.warning("无模型信息，无法训练适应度预测器")
            return False
        
        try:
            # 使用滑动窗口（最近400个样本）
            window_size = min(n, 400)
            raw_chroms = self.history['chromosomes'][-window_size:]
            raw_fitness = np.array(self.history['fitnesses'][-window_size:])
            
            # 过滤掉失败的样本(-inf)
            valid_mask = raw_fitness != -np.inf
            if np.sum(valid_mask) < self.min_samples_to_train:
                return False
            
            valid_chroms = [raw_chroms[i] for i in range(len(raw_chroms)) if valid_mask[i]]
            valid_fitness = raw_fitness[valid_mask]
            
            # 特征工程：将染色体解码为有意义的特征
            X = np.array([self._extract_features(c) for c in valid_chroms])
            y = valid_fitness
            
            # 标准化特征（主要影响超参数值部分，one-hot部分影响较小）
            X_scaled = self.scaler.fit_transform(X)
            
            # 使用交叉验证估计真实预测能力
            try:
                n_samples = len(X_scaled)
                n_cv = min(5, max(2, n_samples // 10))  # 自适应折数
                if n_cv >= 2 and n_samples >= 15:
                    cv_scores = cross_val_score(
                        self.model, X_scaled, y, cv=n_cv, scoring='r2'
                    )
                    # 使用中位数而非均值（对异常折更鲁棒）
                    self.prediction_confidence = max(0.0, float(np.median(cv_scores)))
                else:
                    self.prediction_confidence = 0.0
            except Exception as e:
                logging.debug(f"交叉验证失败: {e}")
                self.prediction_confidence = 0.0
            
            # 训练最终模型（使用全部数据）
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self._samples_since_train = 0
            
            # 清除特征缓存（下次预测会重新计算）
            self._feature_cache.clear()
            
            logging.info(f"✓ 适应度预测模型训练完成 (R²={self.prediction_confidence:.4f}, "
                        f"样本数={n_samples}, 特征维度={X.shape[1]})")
            
            return True
        except Exception as e:
            logging.error(f"适应度预测模型训练失败: {str(e)}")
            self.is_trained = False
            return False
        
    def predict(self, chromosome):
        """预测适应度"""
        if not self.is_trained or self.n_models == 0:
            return None
        
        try:
            features = self._extract_features(chromosome)
            X = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(X)[0]
            return float(prediction)
        except Exception as e:
            logging.warning(f"适应度预测失败: {str(e)}")
            return None
    
    def predict_with_uncertainty(self, chromosome):
        """预测适应度并估计不确定性（使用RF的树预测方差）
        
        :return: (prediction, uncertainty) or (None, None)
        """
        if not self.is_trained or self.n_models == 0:
            return None, None
        
        try:
            features = self._extract_features(chromosome)
            X = self.scaler.transform(features.reshape(1, -1))
            
            # 获取每棵树的预测
            tree_predictions = np.array([
                tree.predict(X)[0] for tree in self.model.estimators_
            ])
            
            prediction = float(np.mean(tree_predictions))
            uncertainty = float(np.std(tree_predictions))
            
            return prediction, uncertainty
        except Exception as e:
            logging.warning(f"适应度预测失败: {str(e)}")
            return None, None
    
    def get_confidence(self):
        """获取预测置信度（基于交叉验证R²）"""
        return self.prediction_confidence if self.is_trained else 0.0

import random
import logging
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import load_data as ld
import normalizer as nl
import imputer as imp
import outlier_detector as out
import duplicate_detector as dup
import feature_selector as fs
import regressor as rg
import classifier as cl
import time
import matplotlib.pyplot as plt
import codecs
import sys
import encoder as enc
import pandas as pd
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    filename='autoML.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 清空日志文件
with open('autoML.log', 'w', encoding='utf-8') as f:
    f.write('')

# 确保控制台输出也支持中文
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

class FitnessPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.history = {
            'chromosomes': [],
            'fitnesses': []
        }
        
    def add_history(self, chromosome, fitness):
        """添加历史数据"""
        self.history['chromosomes'].append(chromosome)
        self.history['fitnesses'].append(fitness)
        
    def train(self):
        """训练预测模型"""
        if len(self.history['chromosomes']) < 100:  # 至少需要100个样本
            return False
            
        X = np.array(self.history['chromosomes'])
        y = np.array(self.history['fitnesses'])
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X, y)
        self.is_trained = True
        return True
        
    def predict(self, chromosome):
        """预测适应度"""
        if not self.is_trained:
            return None
            
        X = np.array([chromosome])
        X = self.scaler.transform(X)
        return self.model.predict(X)[0]

class GeneticAlgorithm:
    def __init__(self, data, target, use_prediction=True, cv_scoring=None, enable_ensemble=False):
        """
        初始化遗传算法
        :param data: 输入数据
        :param target: 目标变量
        :param use_prediction: 是否使用预测模型
        :param cv_scoring: 交叉验证评分方法
        :param enable_ensemble: 是否启用集成
        """
        self.data = data
        self.target = target
        self.use_prediction = use_prediction
        self.enable_ensemble = enable_ensemble
        self.top_models = []  # 存储表现最好的模型
        self.ensemble_size = 5 # 集成模型数量
        
        # 检测任务类型
        self.task_type, target_encoder = self.detect_task_type(data, target)
        logging.info(f"检测到任务类型: {self.task_type}")
        
        # 初始化全局编码器（在进化开始前训练一次）
        logging.info("初始化全局编码器...")
        enc.global_encoder.train_on_data(data, ratio_threshold=0.5, count_threshold=50)
        encoding_summary = enc.global_encoder.get_encoding_summary()
        logging.info(f"全局编码器初始化完成: 编码{encoding_summary['total_features']}个特征，跳过{encoding_summary['skipped_features']}个特征")
        
        # 根据任务类型设置搜索空间
        self.preprocessing_steps = self._get_default_preprocessing_steps()
        self.model_hyperparameters = self._get_default_model_hyperparameters()
        self.pre_len = len(self.preprocessing_steps)
        
        # 设置交叉验证评分方法
        if cv_scoring is None:
            self.cv_scoring = 'neg_mean_squared_error' if self.task_type == 'regression' else 'accuracy'
        else:
            self.cv_scoring = cv_scoring
        self.target_encoder = target_encoder
        
        # 初始化其他属性
        self.best_score = -np.inf
        self.fitness_predictor = FitnessPredictor()

    def detect_task_type(self, data, target):
        """
        自动检测任务类型
        :param data: 输入数据
        :param target: 目标变量
        :return: (任务类型, 目标编码器)
        """
        try:
            # 输入验证
            if data is None or data.empty or target not in data.columns:
                raise ValueError("数据为空或目标变量不存在")
            
            target_series = data[target].dropna()
            if len(target_series) == 0:
                raise ValueError("目标变量没有有效值")
            
            target_dtype = target_series.dtype
            n_unique = target_series.nunique()
            n_samples = len(target_series)
            
            logging.info(f"目标变量 '{target}': 类型={target_dtype}, 唯一值={n_unique}, 样本数={n_samples}")
            
            # 字符串/分类类型 - 需要编码
            if target_dtype in ['object', 'category', 'string']:
                from sklearn.preprocessing import LabelEncoder
                target_encoder = LabelEncoder()
                target_encoder.fit(target_series)
                logging.info(f"已编码目标变量，映射: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
        
                if n_unique == 2:
                    return 'binary_classification', target_encoder
                else:
                    return 'multiclass_classification', target_encoder
            
            # 数值类型判断
            elif target_dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                # 二分类
                if n_unique == 2:
                    return 'binary_classification', None
                
                # 多分类（类别数较少）
                elif n_unique <= 8 and n_unique <= n_samples * 0.1:
                    return 'multiclass_classification', None
                
                # 回归任务
                else:
                    return 'regression', None
            
            # 其他类型默认为回归
            else:
                logging.warning(f"不支持的数据类型: {target_dtype}，默认为回归任务")
                return 'regression', None
                
        except Exception as e:
            logging.error(f"任务类型检测失败: {str(e)}")
            return 'regression', None

    def _get_default_preprocessing_steps(self):
        if self.task_type == 'regression':
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'L1', 'IMP'],
            }
        elif self.task_type == 'binary_classification':
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'Tree', 'WR'],
            }
        else:  # multiclass_classification
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'Tree', 'WR'],
            }

    def _get_default_model_hyperparameters(self):
        if self.task_type == 'regression':
            return {
                'LASSO': {'alpha': [0.0001, 0.001, 0.01, 0.1]},
                'OLS': {},
                'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
                'ElasticNet': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]},
                'SVR': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto', 0.1, 0.01]},
                'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
                'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05], 'max_depth': [3, 4]}
            }
        elif self.task_type == 'binary_classification':
            return {
                'LogisticRegression': {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs'], 'max_iter': [100, 200, 300, 400]},
                'SVM': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
                'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
                'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05], 'max_depth': [3, 4]},
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                'LightGBM': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 63, 127],
                    'subsample': [0.6, 0.8, 1.0]
                }
            }
        else:  # multiclass_classification
            return {
                'LogisticRegression': {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'newton-cg'], 'max_iter': [100, 200, 300, 400]},
                'SVM': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
                'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
                'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05], 'max_depth': [3, 4]},
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                'LightGBM': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 63, 127],
                    'subsample': [0.6, 0.8, 1.0]
                },
                'CatBoost': {
                    'iterations': [50, 100, 200],
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5],
                    'bootstrap_type': ['Bernoulli', 'Bayesian']
                }
            }

    def initialize_population(self, population_size):
        models = list(self.model_hyperparameters.keys())
        max_hyper = max(len(p) for p in self.model_hyperparameters.values()) if self.model_hyperparameters else 0
        population = []
        
        logging.info(f"初始化种群: 模型数量={len(models)}, 最大超参数数量={max_hyper}")
        logging.info(f"可用模型: {models}")
        
        for _ in range(population_size):
            # 确保生成的索引在有效范围内
            chromo = []
            
            # 添加预处理步骤索引
            for step_name, strategies in self.preprocessing_steps.items():
                if len(strategies) > 0:
                    chromo.append(random.randint(0, len(strategies)-1))
                else:
                    chromo.append(0)
            
            # 添加模型索引
            model_idx = random.randint(0, len(models)-1) if models else 0
            chromo.append(model_idx)
            
            # 添加超参数索引
            if models and model_idx < len(models):
                model_name = models[model_idx]
                hyper_params = self.model_hyperparameters[model_name]
                
                # 为当前模型的每个超参数添加索引
                for param_name, values in hyper_params.items():
                    if len(values) > 0:
                        chromo.append(random.randint(0, len(values)-1))
                    else:
                        chromo.append(0)
                
                # 填充剩余位置到最大超参数数量
                remaining_params = max_hyper - len(hyper_params)
                chromo += [0] * remaining_params
            else:
                # 如果没有模型，填充默认值
                chromo += [0] * max_hyper
                
            population.append(chromo)
        return population

    def crossover(self, parent1, parent2):
        same_model = (parent1[self.pre_len] == parent2[self.pre_len])
        if same_model:
            point = random.randint(1, len(parent1)-1)
        else:
            point = random.randint(1, self.pre_len-1)
        return (
            parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:],
        )

    def mutate(self, chromosome, mutation_rate):
        models = list(self.model_hyperparameters.keys())
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                if i < self.pre_len:
                    strategies = list(self.preprocessing_steps.values())[i]
                    if len(strategies) > 0:
                        chromosome[i] = random.randint(0, len(strategies)-1)
                elif i == self.pre_len:
                    if len(models) > 0:
                        new_model = random.randint(0, len(models)-1)
                        chromosome[i] = new_model
                        model_name = models[new_model]
                        params = self.model_hyperparameters[model_name]
                        param_keys = list(params.keys())
                        for j in range(len(param_keys)):
                            pos = self.pre_len + 1 + j
                            if pos < len(chromosome):
                                param_values = params[param_keys[j]]
                                if len(param_values) > 0:
                                    chromosome[pos] = random.randint(0, len(param_values)-1)
                elif i > self.pre_len:
                    model_idx = chromosome[self.pre_len]
                    if model_idx < len(models):
                        model_name = models[model_idx]
                        param_idx = i - (self.pre_len + 1)
                        param_keys = list(self.model_hyperparameters[model_name].keys())
                        if param_idx < len(param_keys):
                            param_key = param_keys[param_idx]
                            param_values = self.model_hyperparameters[model_name][param_key]
                            if len(param_values) > 0:
                                chromosome[i] = random.randint(0, len(param_values)-1)
        return chromosome

    def decode_chromosome(self, chromosome):
        models = list(self.model_hyperparameters.keys())
        decoded = {
            'preprocessing': {},
            'model': None,
            'hyperparameters': {}
        }
        
        # 解码预处理步骤
        index = 0
        for step, strategies in self.preprocessing_steps.items():
            if index < len(chromosome) and len(strategies) > 0:
                strategy_idx = min(chromosome[index], len(strategies)-1)
                decoded['preprocessing'][step] = strategies[strategy_idx]
            else:
                decoded['preprocessing'][step] = strategies[0] if strategies else ''
            index += 1
        
        # 解码模型
        if index < len(chromosome) and len(models) > 0:
            model_idx = min(chromosome[index], len(models)-1)
            decoded['model'] = models[model_idx]
            logging.debug(f"解码模型: 染色体索引={chromosome[index]}, 模型索引={model_idx}, 模型名称={decoded['model']}")
        else:
            decoded['model'] = models[0] if models else None
            logging.debug(f"解码模型: 使用默认模型={decoded['model']}")
        index += 1
            
        # 解码超参数
        if decoded['model'] and decoded['model'] in self.model_hyperparameters:
            model_params = self.model_hyperparameters[decoded['model']]
            param_keys = list(model_params.keys())
            for param_key in param_keys:
                values = model_params[param_key]
                if index < len(chromosome) and len(values) > 0:
                    value_idx = min(chromosome[index], len(values)-1)
                    decoded['hyperparameters'][param_key] = values[value_idx]
                else:
                    decoded['hyperparameters'][param_key] = values[0] if values else None
                index += 1
            
        return decoded
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """
        锦标赛选择
        :param population: 种群
        :param fitnesses: 适应度值列表
        :param tournament_size: 锦标赛规模
        :return: 选择后的种群
        """
        selected = []
        for _ in range(len(population)):
            # 随机选择tournament_size个个体
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitnesses[i] for i in tournament_idx]
            # 选择适应度最高的个体
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected

    def execute_preprocessing_plan(self, data, target, preprocessing_plan):
        """
        执行预处理计划
        
        Parameters:
        -----------
        data : pd.DataFrame
            输入数据
        target : str
            目标变量名
        preprocessing_plan : dict
            预处理计划配置
            
        Returns:
        --------
        pd.DataFrame
            预处理后的数据
        """
        try:
            # 创建数据副本，避免修改原始数据
            X = data.copy()
            
            # 记录预处理步骤
            preprocessing_log = []
            
            # 1. 缺失值处理 - 优先处理，因为其他步骤可能依赖完整数据
            if 'imputer' in preprocessing_plan and preprocessing_plan['imputer']:
                try:
                    logging.info(f"执行缺失值处理: {preprocessing_plan['imputer']}")
                    X = imp.Imputer(X, strategy=preprocessing_plan['imputer']).transform()
                    preprocessing_log.append(f"缺失值处理: {preprocessing_plan['imputer']}")
                except Exception as e:
                    logging.error(f"缺失值处理失败: {str(e)}")
                    # 如果缺失值处理失败，使用简单的前向填充
                    X = X.fillna(method='ffill').fillna(method='bfill')
                    preprocessing_log.append("缺失值处理: 使用前向填充")
            
            # 2. 异常值检测和处理
            if 'outliers' in preprocessing_plan and preprocessing_plan['outliers']:
                try:
                    logging.info(f"执行异常值检测: {preprocessing_plan['outliers']}")
                    X = out.Outlier_detector(X, strategy=preprocessing_plan['outliers'], threshold=0.8).transform()
                    preprocessing_log.append(f"异常值处理: {preprocessing_plan['outliers']}")
                except Exception as e:
                    logging.error(f"异常值检测失败: {str(e)}")
                    preprocessing_log.append("异常值处理: 跳过")
            
            # 3. 重复值检测和处理
            if 'duplicate_detector' in preprocessing_plan and preprocessing_plan['duplicate_detector']:
                try:
                    logging.info(f"执行重复值检测: {preprocessing_plan['duplicate_detector']}")
                    original_size = len(X)
                    X = dup.Duplicate_detector(X, strategy=preprocessing_plan['duplicate_detector']).transform()
                    if X.empty:
                        logging.warning("重复值处理后数据为空，使用原始数据")
                        X = data.copy()
                    else:
                        removed_count = original_size - len(X)
                        preprocessing_log.append(f"重复值处理: {preprocessing_plan['duplicate_detector']} (移除{removed_count}行)")
                except Exception as e:
                    logging.error(f"重复值检测失败: {str(e)}")
                    preprocessing_log.append("重复值处理: 跳过")
            
            # 4. 特征编码 - 处理分类变量
            try:
                logging.info("执行特征编码")
                X = enc.Encoder(dataset=X, ratio_threshold=0.5, count_threshold=50).transform()
                preprocessing_log.append("特征编码: 自动编码")
            except Exception as e:
                logging.error(f"特征编码失败: {str(e)}")
                preprocessing_log.append("特征编码: 跳过")
            
            # 5. 数据标准化/归一化
            if 'normalizer' in preprocessing_plan and preprocessing_plan['normalizer']:
                try:
                    logging.info(f"执行数据标准化: {preprocessing_plan['normalizer']}")
                    X = nl.Normalizer(X, strategy=preprocessing_plan['normalizer']).transform()
                    preprocessing_log.append(f"数据标准化: {preprocessing_plan['normalizer']}")
                except Exception as e:
                    logging.error(f"数据标准化失败: {str(e)}")
                    preprocessing_log.append("数据标准化: 跳过")
            
            # 6. 特征选择 - 最后执行，因为需要目标变量
            if 'feature_selector' in preprocessing_plan and preprocessing_plan['feature_selector']:
                try:
                    logging.info(f"执行特征选择: {preprocessing_plan['feature_selector']}")
                    original_features = len(X.columns)
                    X = fs.Feature_selector(
                        X, 
                        target=target, 
                        strategy=preprocessing_plan['feature_selector'], 
                        threshold=0.1, 
                        exclude=target
                    ).transform()
                    selected_features = len(X.columns)
                    removed_features = original_features - selected_features
                    preprocessing_log.append(f"特征选择: {preprocessing_plan['feature_selector']} (移除{removed_features}个特征)")
                except Exception as e:
                    logging.error(f"特征选择失败: {str(e)}")
                    preprocessing_log.append("特征选择: 跳过")
            
            # 最终数据验证
            if X.empty:
                logging.error("预处理后数据为空，返回原始数据")
                return data.copy()
            
            # 检查是否包含目标变量
            if target not in X.columns:
                logging.warning(f"目标变量 {target} 不在预处理后的数据中，添加回数据")
                X[target] = data[target]
            
            # 记录预处理摘要
            logging.info(f"预处理完成: {len(preprocessing_log)} 个步骤")
            for step in preprocessing_log:
                logging.info(f"  - {step}")
            
            return X
            
        except Exception as e:
            logging.error(f"预处理执行失败: {str(e)}")
            # 返回原始数据，确保算法可以继续运行
            return data.copy()

    def fitness_function(self, chromosome, data, target):
        try:
            config = self.decode_chromosome(chromosome)
            processed_data = self.execute_preprocessing_plan(data.copy(), target, config['preprocessing'])                   
            if self.task_type == 'regression':
                score, model = rg.Regressor(
                    dataset=processed_data,
                    target=target,
                    strategy=config['model'],
                    hyperparameters=config['hyperparameters'],
                    cv_scoring=self.cv_scoring
                ).transform()
                
                if self.enable_ensemble and model is not None and score > -np.inf:
                    self.add_top_models(chromosome, score, model, processed_features=list(processed_data.columns))
                    
                return score, model
            else:  # classification tasks
                score, model = cl.Classifier(
                    dataset=processed_data,
                    target=target,
                    strategy=config['model'],
                    hyperparameters=config['hyperparameters'],
                    cv_scoring=self.cv_scoring
                ).transform()
                
                # 如果启用集成且模型训练成功，添加到集成候选列表
                if self.enable_ensemble and model is not None and score > -np.inf:
                    self.add_top_models(chromosome, score, model, processed_features=list(processed_data.columns))        
                return score, model
        except Exception as e:
            logging.error(f"Fitness calculation error: {str(e)}")
            return -np.inf, None
    
    def evaluate_individual(self, chromo, data, target):
        """评估单个个体的适应度"""
        try:
            if self.use_prediction and self.fitness_predictor.is_trained:
                predicted_fitness = self.fitness_predictor.predict(chromo)
                if predicted_fitness is not None:
                    # 如果预测适应度高于当前最佳适应度的90%，进行实际计算
                    if predicted_fitness > self.best_score * 0.90:
                        actual_fitness, model = self.fitness_function(chromo, data.copy(), target)
                        self.fitness_predictor.add_history(chromo, actual_fitness)
                        return (actual_fitness, chromo, model)
                    # 否则根据阈值决定是否使用预测值
                    elif random.random() < self.prediction_threshold:
                        return (predicted_fitness, chromo, None)           
            # 计算实际适应度
            actual_fitness, model = self.fitness_function(chromo, data.copy(), target)
            self.fitness_predictor.add_history(chromo, actual_fitness)
            if self.use_prediction:        
                if len(self.fitness_predictor.history['chromosomes']) % 300 == 0:
                    logging.info("开始重新训练预测模型...")
                    self.fitness_predictor.train()
                    logging.info("预测模型训练完成")
                
            return (actual_fitness, chromo, model)
            
        except Exception as e:
            logging.error(f"评估错误: {str(e)}")
            return (-np.inf, chromo, None)
    
    def run(self, generations=20, population_size=50, mutation_rate=0.2, 
            elite_size=2, n_jobs=-1, time_limit=None, tournament_size=3):
        """
        运行遗传算法
        :param generations: 迭代次数
        :param population_size: 种群大小
        :param mutation_rate: 变异率
        :param elite_size: 精英个体数量
        :param n_jobs: 并行数
        :param time_limit: 时间限制（秒），如果为None则不限制时间
        :param tournament_size: 锦标赛规模
        :return: (best_config, best_score, history, avg_history, best_model, ensemble_model)
        """

        self.pre_len = len(self.preprocessing_steps)
        data=self.data
        target=self.target
        population = self.initialize_population(population_size)
        best_config = None
        best_score = -np.inf
        best_model = None
        history = []
        avg_history = []
        
        start_time = time.time()
        
        for gen in range(generations):
            # 检查时间限制
            if time_limit is not None and time.time() - start_time > time_limit:
                logging.info(f"达到时间限制 {time_limit} 秒，提前结束进化")
                break
                
            # 并行评估种群适应度
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self.evaluate_individual)(chromo, data, target) 
                for chromo in population
                )
        
            scores = []
            valid_pop = []
            for score, chromo, model in results:
                scores.append(score)
                valid_pop.append(chromo)
                if score > best_score:
                    best_score = score
                    best_config = chromo
                    best_model = model
            
            # 直接使用原始适应度值
            self.best_score = best_score
            valid_scores = [s for s in scores if s != -np.inf]
        
            if valid_scores:
                avg_fitness = np.mean(valid_scores)
            else:
                avg_fitness = -np.inf
            
            avg_history.append(avg_fitness)
            
            # 选择精英个体
            elite = sorted(zip(scores, valid_pop), key=lambda x: x[0], reverse=True)[:elite_size]
            elite = [c for (s, c) in elite]

            # 使用锦标赛选择
            selected_population = self.tournament_selection(valid_pop, scores, tournament_size)

            # 交叉和变异生成子代
            next_gen = []
            while len(next_gen) < population_size - elite_size:
                #交叉逻辑需要改进
                p1, p2 = random.sample(selected_population, 2)
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutate(c1, mutation_rate))
                next_gen.append(self.mutate(c2, mutation_rate))
            
            population = next_gen[:population_size - elite_size] + elite
            
            # 直接使用原始适应度值记录历史
            history.append(best_score)
                
            elapsed_time = time.time() - start_time
            logging.info(f"Generation {gen+1} | Best: {history[-1]:.4f} | Avg: {avg_fitness:.4f} | Time: {elapsed_time:.2f}s")

        
        return best_config, best_score, history, avg_history, best_model

    def add_top_models(self, chromosome, score, model, processed_features=None):
        """
        维护ensemble_size大小的top_models，保证其中模型类型各不相同且分数最优。
        :param chromosome: 染色体（配置）
        :param score: 模型得分
        :param model: 训练好的模型
        :param processed_features: 预处理后的特征列名列表
        """
        # 解码染色体获取模型类型
        config = self.decode_chromosome(chromosome)
        model_type = config['model']
        model_info = {
            'chromosome': chromosome,
            'score': score,
            'model': model,
            'config': config,
            'model_name': f"model_{len(self.top_models)}_{model_type}",
            'estimator_type': 'regressor' if self.task_type == 'regression' else 'classifier',
            'processed_features': processed_features  # 保存预处理后的特征列名
        }
        # 查找是否有相同模型类型
        same_type_idx = next((i for i, m in enumerate(self.top_models) if m['config']['model'] == model_type), None)
        if same_type_idx is not None:
            # 有相同模型类型，若新分数更高则替换
            if score > self.top_models[same_type_idx]['score']:
                self.top_models[same_type_idx] = model_info
        else:
            if len(self.top_models) < self.ensemble_size:
                self.top_models.append(model_info)
            else:
                # 没有相同类型且已满，与最低分比
                min_idx = min(range(len(self.top_models)), key=lambda i: self.top_models[i]['score'])
                if score > self.top_models[min_idx]['score']:
                    self.top_models[min_idx] = model_info
        # 按分数降序排序
        self.top_models.sort(key=lambda x: x['score'], reverse=True)
        logging.info(f"模型已添加到集成候选列表，当前候选模型数量: {len(self.top_models)}")

    def integrated_predict(self, X, target=None):
        """
        对self.top_models中的模型进行投票/加权平均集成
        :param X: 输入数据
        :param target: 目标变量名
        :return: 集成预测结果
        """
        if len(self.top_models) == 0:
            logging.error("集成模型未训练或没有模型")
            return None
        
        # 输出top_models中模型的配置信息
        logging.info("集成模型配置信息:")
        for i, model_info in enumerate(self.top_models):
            config = model_info['config']
            score = model_info['score']
            model_name = model_info['model_name']
            logging.info(f"  模型 {i+1}: {model_name}")
            logging.info(f"    得分: {score:.4f}")
            logging.info(f"    模型类型: {config['model']}")
            logging.info(f"    预处理配置:")
            for key, value in config.items():
                if key != 'model':
                    logging.info(f"      {key}: {value}")
            logging.info("")
        
        try:
            predictions = []
            weights = []
            target_processed = None
            
            # 为每个模型使用其对应的预处理配置
            logging.info("为每个模型使用对应的预处理配置...")
            
            for model_info in self.top_models:
                model = model_info['model']
                model_name = model_info['model_name']
                config = model_info['config']
                processed_features = model_info['processed_features']
                
                try:
                    # 使用每个模型对应的预处理配置
                    preprocessing_config = config['preprocessing'].copy()
                    preprocessing_config['feature_selector'] = ''
                    preprocessing_config['outliers'] = ''
                    preprocessing_config['duplicate_detector'] = ''
                    logging.info(f"为模型 {model_name} 执行预处理...")
                    processed_data = self.execute_preprocessing_plan(
                        X.copy(), target, preprocessing_config
                    )
                    
                    # 分离特征和目标变量
                    if target is not None and target in processed_data.columns:
                        target_processed = processed_data[target]
                        processed_X = processed_data.drop(columns=[target])
                        
                        # 如果是分类任务且目标变量被编码了，需要解码
                        if self.task_type != 'regression' and self.target_encoder is not None:
                            try:
                                # 尝试解码目标变量
                                target_processed = self.target_encoder.inverse_transform(target_processed)
                            except:
                                # 如果解码失败，保持原样
                                pass
                    else:
                        processed_X = processed_data
                    
                    logging.info(f"模型 {model_name} 预处理完成，特征数量: {len(processed_X.columns)}")
                    
                    # 确保特征列与训练时一致
                    if processed_features is not None:
                        # 获取训练时的特征列（排除目标变量）
                        expected_features = [f for f in processed_features if f != target]
                        current_features = list(processed_X.columns)
                        
                        # 如果特征不匹配，进行特征对齐
                        if set(expected_features) != set(current_features):
                            logging.warning(f"模型 {model_name} 特征不匹配，进行特征对齐")
                            logging.info(f"   期望特征: {expected_features}")
                            logging.info(f"   当前特征: {current_features}")
                            
                            # 移除多余的特征
                            features_to_remove = [f for f in current_features if f not in expected_features]
                            if features_to_remove:
                                logging.info(f"   移除多余特征: {features_to_remove}")
                                processed_X = processed_X.drop(columns=features_to_remove)
                            
                            # 确保特征顺序一致
                            if len(expected_features) > 0:
                                processed_X = processed_X[expected_features]
                            
                            logging.info(f"   对齐后特征: {list(processed_X.columns)}")
                        
                        # 移除非数值特征
                        non_numeric_features = processed_X.select_dtypes(include=['object', 'string']).columns
                        if len(non_numeric_features) > 0:
                            logging.info(f"   移除非数值特征: {list(non_numeric_features)}")
                            processed_X = processed_X.drop(columns=non_numeric_features)
                        
                        # 确保所有特征都是数值类型
                        processed_X = processed_X.select_dtypes(include=[np.number])
                    
                    # 进行预测
                    pred = model.predict(processed_X)
                    predictions.append(pred)
                    # 使用模型保存的正确率作为权重
                    model_score = model_info['score']
                    weights.append(model_score)
                    
                    logging.info(f"模型 {model_name} 预测完成，预测结果长度: {len(pred)}，权重: {model_score:.4f}")
                    
                except Exception as e:
                    logging.error(f"模型 {model_name} 预测失败: {str(e)}")
                    continue
            
            if not predictions:
                logging.error("所有模型预测都失败了")
                return None
            
            # 验证所有预测结果长度一致
            prediction_lengths = [len(pred) for pred in predictions]
            if len(set(prediction_lengths)) > 1:
                logging.error(f"预测结果长度不一致: {prediction_lengths}")
                return None
            # 计算加权平均
            predictions = np.array(predictions)
            weights = np.array(weights)
            
            # 输出权重信息
            logging.info("模型权重分配:")
            for i, (model_info, weight) in enumerate(zip(self.top_models, weights)):
                logging.info(f"  模型 {i+1} ({model_info['model_name']}): 权重 {weight:.4f}")
            
            # 归一化权重
            weights = weights / np.sum(weights)
            logging.info(f"归一化后权重: {weights}")
            
            final_prediction = np.average(predictions, axis=0, weights=weights)
            
            if self.task_type == 'classification':
                final_prediction = np.round(final_prediction).astype(int)
            
            logging.info(f"集成预测完成，最终预测结果长度: {len(final_prediction)}")
            return final_prediction, target_processed
            
        except Exception as e:
            logging.error(f"集成预测失败: {str(e)}")
            return None

if __name__ == "__main__":
    import load_data as ld
    from sklearn.model_selection import train_test_split
    data = ld.load_data("datasets/titanic_train.csv")
    target = "Survived"
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
 
    ga_ensemble = GeneticAlgorithm(
        data=train_data,
        target=target,
        use_prediction=True, 
        enable_ensemble=True 
    )
    best_config, best_score, history, avg_history, best_model = ga_ensemble.run(
        generations=5,  
        population_size=5,  
        n_jobs=1  # 禁用并行运行，使用单线程
    )
    
    print(f"\n最佳单个模型得分: {best_score:.4f}")
    
    # 最佳单个模型在测试集上的预测
    print("\n=== 最佳单个模型测试 ===")
    if best_model is not None and best_config is not None:
        try:
            # 解码最佳配置
            config = ga_ensemble.decode_chromosome(best_config)
            print(f"最佳模型类型: {config['model']}")
            print(f"最佳模型配置: {config}")
            
            # 对测试集进行相同的预处理
            processed_test_data = ga_ensemble.execute_preprocessing_plan(
                test_data.copy(), target, config['preprocessing']
            )
            
            # 分离特征和目标变量
            if target in processed_test_data.columns:
                y_true = processed_test_data[target]
                X_test = processed_test_data.drop(columns=[target])
            else:
                print("警告: 目标变量不在预处理后的数据中")
                y_true = test_data[target]
                X_test = processed_test_data
            
            # 确保所有特征都是数值类型
            X_test = X_test.select_dtypes(include=[np.number])
            
            # 进行预测
            y_pred = best_model.predict(X_test)
            
            # 评估结果
            if ga_ensemble.task_type == "regression":
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                print(f"最佳单个模型MSE: {mse:.4f}")
                print(f"最佳单个模型R²: {r2:.4f}")
            else:  # classification
                from sklearn.metrics import accuracy_score, classification_report
                accuracy = accuracy_score(y_true, y_pred)
                print(f"最佳单个模型准确率: {accuracy:.4f}")
                print("\n分类报告:")
                print(classification_report(y_true, y_pred))
                
        except Exception as e:
            print(f"最佳单个模型测试失败: {str(e)}")
    else:
        print("没有找到最佳模型")
    
    # 集成预测
    print("\n=== 集成模型测试 ===")
    result = ga_ensemble.integrated_predict(test_data, target)
    if result is not None:
        y_pre, y_true = result
        
        # 确保数据类型匹配
        if ga_ensemble.task_type == "regression":
            from sklearn.metrics import mean_squared_error
            # 确保都是数值类型
            y_true = pd.to_numeric(y_true, errors='coerce').dropna()
            y_pre = pd.to_numeric(y_pre, errors='coerce')
            # 对齐长度
            min_len = min(len(y_true), len(y_pre))
            y_true = y_true.iloc[:min_len]
            y_pre = y_pre[:min_len]
            mse = mean_squared_error(y_true, y_pre)
            print(f"集成模型MSE: {mse:.4f}")
        else:  # classification
            from sklearn.metrics import accuracy_score
            # 确保都是整数类型
            y_true = pd.to_numeric(y_true, errors='coerce').dropna().astype(int)
            y_pre = pd.to_numeric(y_pre, errors='coerce').astype(int)
            # 对齐长度
            min_len = min(len(y_true), len(y_pre))
            y_true = y_true.iloc[:min_len]
            y_pre = y_pre[:min_len]
            accuracy = accuracy_score(y_true, y_pre)
            print(f"集成模型准确率: {accuracy:.4f}")
    else:
        print("集成预测失败")
 

  
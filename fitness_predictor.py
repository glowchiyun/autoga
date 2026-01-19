"""
适应度预测器
使用机器学习模型预测遗传算法个体的适应度，减少昂贵的评估次数
"""
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class FitnessPredictor:
    """适应度预测器：使用随机森林预测染色体的适应度"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.history = {
            'chromosomes': [],
            'fitnesses': []
        }
        self.min_samples_to_train = 50  # 降低到50而非100，使预测模型更早投入使用
        
    def add_history(self, chromosome, fitness):
        """添加历史数据"""
        self.history['chromosomes'].append(chromosome)
        self.history['fitnesses'].append(fitness)
        
    def train(self):
        """训练预测模型 - 改进版本，过滤失败样本并使用最近数据"""
        if len(self.history['chromosomes']) < self.min_samples_to_train:
            return False
        
        try:
            # 为了稳定性，只使用最近的样本（滑动窗口）
            recent_size = min(len(self.history['chromosomes']), 200)
            X = np.array(self.history['chromosomes'][-recent_size:])
            y = np.array(self.history['fitnesses'][-recent_size:])
            
            # 过滤掉失败的样本(-inf)
            valid_mask = y != -np.inf
            if np.sum(valid_mask) < self.min_samples_to_train:
                return False
            
            X = X[valid_mask]
            y = y[valid_mask]
            
            # 标准化特征
            X = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X, y)
            self.is_trained = True
            
            # 记录模型性能
            train_r2 = self.model.score(X, y)
            logging.debug(f"预测模型训练完成，R²={train_r2:.4f}")
            
            return True
        except Exception as e:
            logging.error(f"预测模型训练失败: {str(e)}")
            self.is_trained = False
            return False
        
    def predict(self, chromosome):
        """预测适应度"""
        if not self.is_trained:
            return None
        
        try:
            X = np.array([chromosome])
            X = self.scaler.transform(X)
            prediction = self.model.predict(X)[0]
            return prediction
        except Exception as e:
            logging.error(f"预测失败: {str(e)}")
            return None

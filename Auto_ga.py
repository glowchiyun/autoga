import random
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import codecs
import sys
from scipy import stats
import encoder as enc
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
    def __init__(self, use_prediction=True,cv_scoring='neg_mean_squared_error'):
        self.preprocessing_steps = None
        self.model_hyperparameters = None
        self.pre_len = None
        self.fitness_predictor = FitnessPredictor()
        self.use_prediction = use_prediction
        self.prediction_threshold = 0.9
        self.best_score = -np.inf
        self.task_type = None  
        self.cv_scoring = cv_scoring
    def detect_task_type(self, data, target):
        """
        自动检测任务类型
        :param data: 输入数据
        :param target: 目标变量
        :return: 任务类型 ('regression', 'binary_classification', 'multiclass_classification')
        """
        # 创建数据副本
        data_encoded = data.copy()
        target_encoder = None
        
        # 如果目标变量不是数值类型，先进行编码
        if data[target].dtype == 'object' or data[target].dtype == 'category':
            from sklearn.preprocessing import LabelEncoder
            target_encoder = LabelEncoder()
            data_encoded[target] = target_encoder.fit_transform(data[target])
            logging.info(f"目标变量 '{target}' 已使用LabelEncoder编码")
            logging.info(f"编码映射: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
        
        # 使用编码后的数据检测任务类型
        target_values = data_encoded[target].unique()
        n_classes = len(target_values)
        
        if data_encoded[target].dtype in ['float64', 'float32', 'int64', 'int32']:
            # 检查是否可能是回归任务
            if n_classes > 10:  
                return 'regression', target_encoder
        
        # 分类任务
        if n_classes == 2:
            return 'binary_classification', target_encoder
        else:
            return 'multiclass_classification', target_encoder

    def _get_default_preprocessing_steps(self):
        if self.task_type == 'regression':
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED', 'MARK', 'DL', 'LM', 'JW'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'L1', 'IMP'],
                'encoder': ['',  'LE', 'OHE', 'BE', 'WOE']
            }
        elif self.task_type == 'binary_classification':
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED', 'MARK', 'DL', 'LM', 'JW'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'Tree', 'WR', 'SVC'],
                'encoder': ['',  'LE', 'OHE', 'BE', 'WOE']
            }
        else:  # multiclass_classification
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED', 'MARK', 'DL', 'LM', 'JW'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'Tree', 'WR', 'SVC'],
                'encoder': ['',  'LE', 'OHE', 'BE', 'WOE']
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
        max_hyper = max(len(p) for p in self.model_hyperparameters.values())
        population = []
        for _ in range(population_size):
            chromo = [random.randint(0, len(s)-1) for s in self.preprocessing_steps.values()]
            model_idx = random.randint(0, len(models)-1)
            chromo.append(model_idx)
            model_name = models[model_idx]
            hyper_params = self.model_hyperparameters[model_name]
            chromo += [random.randint(0, len(v)-1) for v in hyper_params.values()]
            chromo += [0]*(max_hyper - len(hyper_params))
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
                    chromosome[i] = random.randint(0, len(strategies)-1)
                elif i == self.pre_len:
                    new_model = random.randint(0, len(models)-1)
                    chromosome[i] = new_model
                    model_name = models[new_model]
                    params = self.model_hyperparameters[model_name]
                    for j in range(len(params)):
                        pos = self.pre_len + 1 + j
                        chromosome[pos] = random.randint(0, len(params[list(params.keys())[j]])-1)
                elif i > self.pre_len:
                    model_idx = chromosome[self.pre_len]
                    model_name = models[model_idx]
                    param_idx = i - (self.pre_len + 1)
                    if param_idx < len(self.model_hyperparameters[model_name]):
                        param = list(self.model_hyperparameters[model_name].values())[param_idx]
                        chromosome[i] = random.randint(0, len(param)-1)
        return chromosome

    def decode_chromosome(self, chromosome):
        models = list(self.model_hyperparameters.keys())
        decoded = {
            'preprocessing': {},
            'model': None,
            'hyperparameters': {}
        }
        index = 0
        for step, strategies in self.preprocessing_steps.items():
            decoded['preprocessing'][step] = strategies[chromosome[index]]
            index += 1
        model_idx = chromosome[index]
        decoded['model'] = models[model_idx]
        index += 1
        model_params = self.model_hyperparameters[decoded['model']]
        for i, (param, values) in enumerate(model_params.items()):
            decoded['hyperparameters'][param] = values[chromosome[index+i]]
        return decoded
    
    #轮盘赌算法
    #def roulette_wheel_selection(self, population, fitnesses, similarities=None):
    #    total_fitness = sum(fitnesses)
    #    selection_probs = [f / total_fitness for f in fitnesses]
    #    selected_indices = np.random.choice(len(population), size=len(population), replace=True, p=selection_probs)
    #    return [population[i] for i in selected_indices]

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
        X = pd.DataFrame()
        if 'imputer' in preprocessing_plan:
            imp1 = imp.Imputer(data.copy(), strategy=preprocessing_plan['imputer']).transform()
            X = imp1
        if 'normalizer' in preprocessing_plan and preprocessing_plan['normalizer'] != '':
            if X.empty:
                X = data.copy()
            n1 = nl.Normalizer(X, strategy=preprocessing_plan['normalizer']).transform()
            X = n1
        if 'outliers' in preprocessing_plan and preprocessing_plan['outliers'] != '':
            if X.empty:
                X = data.copy()
            out1 = out.Outlier_detector(X, strategy=preprocessing_plan['outliers'], threshold=0.3).transform()
            X = out1
        if 'duplicates' in preprocessing_plan and preprocessing_plan['duplicate_detector'] != '':
            if X.empty:
                X = data.copy()
            dup1 = dup.Duplicate_detector(X, strategy=preprocessing_plan['duplicate_detector']).transform()
            X = dup1
        if 'encoder' in preprocessing_plan and preprocessing_plan['encoder'] != '':
            if X.empty:
                X = data.copy()
            enc1 = enc.Encoder(X, strategy=preprocessing_plan['encoder'], target=target).transform()
            X = enc1
        if 'feature_selector' in preprocessing_plan and preprocessing_plan['feature_selector'] != '':
            if X.empty:
                X = data.copy()
            fs1 = fs.Feature_selector(X, target=target, strategy=preprocessing_plan['feature_selector'], threshold=0.1, exclude=target).transform()
            X = fs1
        return X

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
                return score, model
            else:  # classification tasks
                score, model = cl.Classifier(
                    dataset=processed_data,
                    target=target,
                    strategy=config['model'],
                    hyperparameters=config['hyperparameters'],
                    cv_scoring=self.cv_scoring
                ).transform()
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
                    if predicted_fitness > self.best_score * 0.95:
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
    
    def run(self, data, target, generations=20, population_size=50, mutation_rate=0.2, 
            elite_size=2, n_jobs=-1, time_limit=None, tournament_size=3):
        """
        运行遗传算法
        :param data: 输入数据
        :param target: 目标变量
        :param generations: 迭代次数
        :param population_size: 种群大小
        :param mutation_rate: 变异率
        :param elite_size: 精英个体数量
        :param n_jobs: 并行数
        :param time_limit: 时间限制（秒），如果为None则不限制时间
        :param tournament_size: 锦标赛规模
        :return: (best_config, best_score, history, avg_history, best_model)
        """
        # 检测任务类型并获取编码器
        self.task_type, target_encoder = self.detect_task_type(data, target)
        logging.info(f"检测到任务类型: {self.task_type}")
        
        # 如果目标变量被编码，使用编码后的数据
        if target_encoder is not None:
            data = data.copy()
            data[target] = target_encoder.transform(data[target])
        
        # 根据任务类型设置搜索空间
        self.preprocessing_steps = self._get_default_preprocessing_steps()
        self.model_hyperparameters = self._get_default_model_hyperparameters()
        self.pre_len = len(self.preprocessing_steps)
        
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

def adjusted_r2_score(y_true, y_pred, n_features):
    """
    计算调整后的R2分数
    :param y_true: 真实值
    :param y_pred: 预测值
    :param n_features: 特征数量
    :return: 调整后的R2分数
    """
    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n_samples <= n_features + 1:
        return r2
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

if __name__ == "__main__":
    def save_results_to_log(results, config):
        """
        将结果保存到日志文件
        :param results: 测试结果字典
        :param config: 最佳配置
        """
        logging.info("\n" + "="*50)
        logging.info("遗传算法运行结果总结")
        logging.info("="*50)
        
        # 保存时间信息
        logging.info("\n运行时间比较:")
        logging.info(f"不使用预测的平均运行时间: {results['time_no_prediction']:.2f} 秒")
        logging.info(f"使用预测的平均运行时间: {results['time_with_prediction']:.2f} 秒")
        logging.info(f"平均时间节省: {results['time_saving_percentage']:.2f}%")
        
        # 保存预测模型性能
        logging.info("\n预测模型性能:")
        logging.info(f"平均R2分数: {results['prediction_r2']:.4f}")
        logging.info(f"平均调整后R2分数: {results['prediction_adj_r2']:.4f}")
        logging.info(f"平均均方误差: {results['prediction_mse']:.4f}")
        
        # 保存最佳适应度
        logging.info("\n最佳适应度比较:")
        logging.info(f"不使用预测的平均最佳适应度: {results['best_fitness_no_prediction']:.4f}")
        logging.info(f"使用预测的平均最佳适应度: {results['best_fitness_with_prediction']:.4f}")
        
        # 保存最佳配置
        logging.info("\n优化后的管道配置:")
        for step, method in config['preprocessing'].items():
            logging.info(f"{step}: {method}")
        logging.info(f"模型: {config['model']}")
        logging.info("超参数:")
        for param, value in config['hyperparameters'].items():
            logging.info(f"  {param}: {value}")
        
        logging.info("\n" + "="*50)

    def test_fitness_prediction(data, target, generations=10, population_size=20, n_jobs=-1, num_runs=5, time_limit=None, cv_scoring='neg_mean_squared_error'):
        """
        测试适应度预测的性能（多次运行）
        :param data: 输入数据
        :param target: 目标变量
        :param generations: 迭代次数
        :param population_size: 种群大小
        :param n_jobs: 并行数
        :param num_runs: 运行次数
        :param time_limit: 时间限制（秒）
        :return: 性能比较结果
        """
        all_results = []
        history_no_pred_list = []
        history_with_pred_list = []
        
        for run in range(num_runs):
            logging.info(f"\n开始第 {run + 1}/{num_runs} 次运行")
            
            # 1. 不使用预测的运行
            logging.info("运行不使用适应度预测的版本...")
            ga_no_pred = GeneticAlgorithm(use_prediction=False,cv_scoring=cv_scoring)
            start_time = time.time()
            best_chrom_no_pred, best_score_no_pred, history_no_pred, _, _ = ga_no_pred.run(
                data=data,
                target=target,
                generations=generations,
                population_size=population_size,
                n_jobs=n_jobs,
                time_limit=time_limit,
                
            )
            time_no_pred = time.time() - start_time
            history_no_pred_list.append(history_no_pred)
            
            # 2. 使用预测的运行
            logging.info("运行使用适应度预测的版本...")
            ga_with_pred = GeneticAlgorithm(use_prediction=True,cv_scoring=cv_scoring)
            start_time = time.time()
            best_chrom_with_pred, best_score_with_pred, history_with_pred, _, _ = ga_with_pred.run(
                data=data,
                target=target,
                generations=generations,
                population_size=population_size,
                n_jobs=n_jobs,
                time_limit=time_limit,
                
            )
            time_with_pred = time.time() - start_time
            history_with_pred_list.append(history_with_pred)
            
            # 3. 评估预测模型
            logging.info("评估预测模型性能...")
            actual_fitnesses = []
            predicted_fitnesses = []
            
            for chromo, actual_fitness in zip(ga_no_pred.fitness_predictor.history['chromosomes'],
                                            ga_no_pred.fitness_predictor.history['fitnesses']):
                predicted = ga_with_pred.fitness_predictor.predict(chromo)
                if predicted is not None:
                    actual_fitnesses.append(actual_fitness)
                    predicted_fitnesses.append(predicted)
            
            # 计算预测性能指标
            n_features = len(ga_with_pred.fitness_predictor.history['chromosomes'][0])
            r2 = r2_score(actual_fitnesses, predicted_fitnesses)
            adj_r2 = adjusted_r2_score(actual_fitnesses, predicted_fitnesses, n_features)
            mse = mean_squared_error(actual_fitnesses, predicted_fitnesses)
            
            # 计算时间节省
            time_saving = (time_no_pred - time_with_pred) / time_no_pred * 100
            
            # 收集本次运行结果
            run_results = {
                'time_no_prediction': time_no_pred,
                'time_with_prediction': time_with_pred,
                'time_saving_percentage': time_saving,
                'prediction_r2': r2,
                'prediction_adj_r2': adj_r2,
                'prediction_mse': mse,
                'best_fitness_no_prediction': best_score_no_pred,
                'best_fitness_with_prediction': best_score_with_pred,
                'history_no_prediction': history_no_pred,
                'history_with_prediction': history_with_pred
            }
            
            all_results.append(run_results)
            
            logging.info(f"第 {run + 1} 次运行完成:")
            logging.info(f"不使用预测的运行时间: {time_no_pred:.2f}秒")
            logging.info(f"使用预测的运行时间: {time_with_pred:.2f}秒")
            logging.info(f"时间节省: {time_saving:.2f}%")
            logging.info(f"R2分数: {r2:.4f}")
            logging.info(f"调整后的R2分数: {adj_r2:.4f}")
            logging.info(f"均方误差: {mse:.4f}")
        
        # 计算平均结果
        avg_results = {
            'time_no_prediction': np.mean([r['time_no_prediction'] for r in all_results]),
            'time_with_prediction': np.mean([r['time_with_prediction'] for r in all_results]),
            'time_saving_percentage': np.mean([r['time_saving_percentage'] for r in all_results]),
            'prediction_r2': np.mean([r['prediction_r2'] for r in all_results]),
            'prediction_adj_r2': np.mean([r['prediction_adj_r2'] for r in all_results]),
            'prediction_mse': np.mean([r['prediction_mse'] for r in all_results]),
            'best_fitness_no_prediction': np.mean([r['best_fitness_no_prediction'] for r in all_results]),
            'best_fitness_with_prediction': np.mean([r['best_fitness_with_prediction'] for r in all_results])
        }
        
        # 获取最后一次运行的最佳配置
        best_config = ga_with_pred.decode_chromosome(best_chrom_with_pred)
        
        # 保存结果到日志
        save_results_to_log(avg_results, best_config)
        
        return avg_results, all_results

    # Run test
    data = ld.load_data("datasets/titanic_train.csv")
    target = "Embarked"
    avg_results, all_results = test_fitness_prediction(
        data=data,
        target=target,
        generations=20,
        population_size=20,
        num_runs=1,
        cv_scoring='accuracy'
    )


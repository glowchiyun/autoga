import random
import logging
import load_data as ld
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import regressor as rg
import classifier as cl
import time
import matplotlib.pyplot as plt
import codecs
import sys
import pandas as pd
import preprocessing as pre

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
        self.model_hyperparameters = self._get_default_model_hyperparameters()
        
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
                #'SVM': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
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
                #'SVM': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
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
            chromo = []
            # 只保留模型索引和超参数索引
            model_idx = random.randint(0, len(models)-1) if models else 0
            chromo.append(model_idx)

            if models and model_idx < len(models):
                model_name = models[model_idx]
                hyper_params = self.model_hyperparameters[model_name]
                for param_name, values in hyper_params.items():
                    if len(values) > 0:
                        chromo.append(random.randint(0, len(values)-1))
                    else:
                        chromo.append(0)
                # 填充剩余位置到最大超参数数量
                remaining_params = max_hyper - len(hyper_params)
                chromo += [0] * remaining_params
            else:
                chromo += [0] * max_hyper

            population.append(chromo)
        return population

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1)-1)
        return (
            parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:],
        )

    def mutate(self, chromosome, mutation_rate):
        models = list(self.model_hyperparameters.keys())
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                if i == 0:
                    if len(models) > 0:
                        new_model = random.randint(0, len(models)-1)
                        chromosome[i] = new_model
                        model_name = models[new_model]
                        params = self.model_hyperparameters[model_name]
                        param_keys = list(params.keys())
                        for j in range(len(param_keys)):
                            pos = 1 + j
                            if pos < len(chromosome):
                                param_values = params[param_keys[j]]
                                if len(param_values) > 0:
                                    chromosome[pos] = random.randint(0, len(param_values)-1)
                elif i > 0:
                    model_idx = chromosome[0]
                    if model_idx < len(models):
                        model_name = models[model_idx]
                        param_idx = i - 1
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
            'model': None,
            'hyperparameters': {}
        }

        index = 0
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
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitnesses[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected

    def fitness_function(self, chromosome, data, target):
        try:
            config = self.decode_chromosome(chromosome)
            processed_data = data.copy()
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
                    if predicted_fitness > self.best_score * 0.90:
                        actual_fitness, model = self.fitness_function(chromo, data.copy(), target)
                        self.fitness_predictor.add_history(chromo, actual_fitness)
                        return (actual_fitness, chromo, model)
                    elif random.random() < self.prediction_threshold:
                        return (predicted_fitness, chromo, None)
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
            # 交叉时随机选择两个模型相同的染色体进行交叉，需保证该模型在当前种群中至少有两个以上
            # 统计每种模型的染色体索引
            from collections import defaultdict
            model_to_indices = defaultdict(list)
            models = list(self.model_hyperparameters.keys())
            for idx, chromo in enumerate(selected_population):
                if len(chromo) > 0:
                    model_idx = chromo[0]
                    if model_idx < len(models):
                        model_to_indices[model_idx].append(idx)
            # 找到有两个及以上个体的模型
            candidate_model_indices = [m for m, idxs in model_to_indices.items() if len(idxs) >= 2]
            if not candidate_model_indices:
                # 如果没有任何模型有两个及以上的个体，则为每次交叉随机生成一个与现有个体模型类型相同的新个体，与现有个体交叉
                while len(next_gen) < population_size - elite_size:
                    # 随机选一个现有个体
                    p1 = random.choice(selected_population)
                    # 获取其模型类型
                    model_idx = p1[0] if len(p1) > 0 else 0
                    # 随机生成一个模型类型相同的新个体
                    p2 = p1.copy()
                    # 除了模型类型外，其他基因随机生成
                    for i in range(1, len(p2)):
                        strategies = list(self.model_hyperparameters.values())[i-1] if i-1 < len(self.model_hyperparameters) else []
                        if strategies:
                            p2[i] = random.randint(0, len(strategies)-1)
                    c1, c2 = self.crossover(p1, p2)
                    next_gen.append(self.mutate(c1, mutation_rate))
                    next_gen.append(self.mutate(c2, mutation_rate))
            else:
                while len(next_gen) < population_size - elite_size:
                    # 随机选一个有两个及以上个体的模型
                    model_idx = random.choice(candidate_model_indices)
                    idxs = model_to_indices[model_idx]
                    # 随机选两个不同的个体
                    p1_idx, p2_idx = random.sample(idxs, 2)
                    p1 = selected_population[p1_idx]
                    p2 = selected_population[p2_idx]
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

        try:
            predictions = []
            weights = []

            for model_info in self.top_models:
                model = model_info['model']
                model_name = model_info['model_name']
                try:
                    X_input = X.copy()
                    if self.target in X_input.columns:
                        X_input = X_input.drop(columns=[self.target])
                    pred = model.predict(X_input)
                    predictions.append(pred)
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

            predictions = np.array(predictions)
            weights = np.array(weights)

            # 归一化权重
            weights = weights / np.sum(weights)

            final_prediction = np.average(predictions, axis=0, weights=weights)
            final_prediction = np.array(final_prediction).flatten()  # 保证为一维

            if self.task_type == 'binary_classification' or self.task_type == 'multiclass_classification':
                final_prediction = np.round(final_prediction).astype(int)

            logging.info(f"集成预测完成，最终预测结果长度: {len(final_prediction)}")
            return final_prediction

        except Exception as e:
            logging.error(f"集成预测失败: {str(e)}")
            return None



if __name__ == "__main__":
    import load_data as ld
    from sklearn.model_selection import train_test_split
    data = ld.load_data("datasets/titanic_train.csv")
    target = "Survived"
    Pre = pre.Preprocessing(data=data, target=target)
    Pre.run()
    data=Pre.get_processed_data()
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)
    # 2. 用处理后的数据集做模型/超参数搜索
    ga_ensemble = GeneticAlgorithm(
        data=data,
        target=target,
        use_prediction=True, 
        enable_ensemble=True 
    )
    best_config, best_score, history, avg_history, best_model = ga_ensemble.run(
        generations=20,  
        population_size=20,  
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
            
            # 分离特征和目标变量
            if target in test_data.columns:
                y_true = test_data[target]
                X_test = test_data.drop(columns=[target])
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
        y_pre, y_true = result,test_data[target]
        
        # 确保数据类型匹配
        if ga_ensemble.task_type == "regression":
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_true, y_pre)
            print(f"集成模型MSE: {mse:.4f}")
        else:  # classification
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true, y_pre)
            print(f"集成模型准确率: {accuracy:.4f}")
    else:
        print("集成预测失败")
 

  
import random
import logging
import load_data as ld
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
import regressor as rg
import classifier as cl
import time
import matplotlib.pyplot as plt
import codecs
import sys
import os
import pandas as pd
import preprocessing as pre
from range_adjuster import IntelligentRangeAdjuster
from fitness_predictor import FitnessPredictor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 全局日志控制
VERBOSE_MODE = os.environ.get('AUTOML_VERBOSE', 'FALSE').upper() == 'TRUE'
LOG_LEVEL = logging.DEBUG if VERBOSE_MODE else logging.INFO

# 自定义过滤器，减少重复日志
class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = None
        self.repeat_count = 0
        
    def filter(self, record):
        current_log = (record.levelname, record.msg % record.args if record.args else record.msg)
        if current_log == self.last_log:
            self.repeat_count += 1
            return False
        else:
            self.last_log = current_log
            self.repeat_count = 0
            return True

# 配置日志
logging.basicConfig(
    level=LOG_LEVEL,
    filename='autoML.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 为根日志器添加过滤器
for handler in logging.root.handlers:
    handler.addFilter(DuplicateFilter())

# 清空日志文件
with open('autoML.log', 'w', encoding='utf-8') as f:
    f.write('')

# 导入日志配置
try:
    from log_config import setup_logging, suppress_print
    setup_logging('autoML.log')
except ImportError:
    # 如果log_config不存在，使用基本配置
    pass


class GeneticAlgorithm:
    def __init__(self, data, target, use_prediction=True, cv_scoring=None, enable_ensemble=False, enable_adaptive_range=True):
        """
        初始化遗传算法
        :param data: 输入数据
        :param target: 目标变量
        :param use_prediction: 是否使用预测模型
        :param cv_scoring: 交叉验证评分方法
        :param enable_ensemble: 是否启用集成
        :param enable_adaptive_range: 是否启用自适应超参数范围调整
        """
        self.data = data
        self.target = target
        self.use_prediction = use_prediction
        self.enable_ensemble = enable_ensemble
        self.enable_adaptive_range = enable_adaptive_range  # 新增
        self.top_models = []  # 存储表现最好的模型
        self.ensemble_size = 5 # 集成模型数量
        self.best_model_features = None  # 保存最佳模型的特征列
        self.prediction_threshold = 0.8  # 预测适应度阈值
        
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
        self.range_adjuster = IntelligentRangeAdjuster() if enable_adaptive_range else None  # 新增

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
            
            logging.debug(f"目标变量 '{target}': 类型={target_dtype}, 唯一值={n_unique}, 样本数={n_samples}")
            
            # 字符串/分类类型 - 需要编码
            if target_dtype in ['object', 'category', 'string']:
                from sklearn.preprocessing import LabelEncoder
                target_encoder = LabelEncoder()
                target_encoder.fit(target_series)
                logging.debug(f"已编码目标变量，映射: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
        
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

        logging.debug(f"初始化种群: 模型数量={len(models)}, 最大超参数数量={max_hyper}")
        logging.debug(f"可用模型: {models}")

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
        """修复版本：正确处理模型切换时的超参数重新初始化"""
        models = list(self.model_hyperparameters.keys())
        if not models:
            return chromosome
        
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                if i == 0:  # 模型索引位置
                    old_model_idx = chromosome[i]
                    new_model_idx = random.randint(0, len(models) - 1)
                    
                    if new_model_idx != old_model_idx:
                        # 模型类型改变 - 需要重新初始化所有超参数
                        chromosome[i] = new_model_idx
                        new_model_name = models[new_model_idx]
                        new_params = self.model_hyperparameters[new_model_name]
                        
                        # 重置所有超参数基因
                        for j in range(1, len(chromosome)):
                            param_keys = list(new_params.keys())
                            if j - 1 < len(param_keys):
                                param_values = new_params[param_keys[j - 1]]
                                chromosome[j] = random.randint(0, len(param_values) - 1) \
                                               if param_values else 0
                            else:
                                chromosome[j] = 0
                        
                        logging.debug(f"模型变异: {models[old_model_idx]} -> {new_model_name}")
                    else:
                        chromosome[i] = new_model_idx
                        
                else:  # 超参数位置
                    model_idx = chromosome[0]
                    if model_idx < len(models):
                        model_name = models[model_idx]
                        param_keys = list(self.model_hyperparameters[model_name].keys())
                        param_idx = i - 1
                        
                        if param_idx < len(param_keys):
                            param_key = param_keys[param_idx]
                            param_values = self.model_hyperparameters[model_name][param_key]
                            if param_values:
                                chromosome[i] = random.randint(0, len(param_values) - 1)
        
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
                    # 保存训练时使用的特征列（去除目标变量）
                    train_features = [col for col in processed_data.columns if col != target]
                    self.add_top_models(chromosome, score, model, processed_features=train_features)

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
                    # 保存训练时使用的特征列（去除目标变量）
                    train_features = [col for col in processed_data.columns if col != target]
                    self.add_top_models(chromosome, score, model, processed_features=train_features)
                return score, model
        except Exception as e:
            logging.error(f"Fitness calculation error: {str(e)}")
            return -np.inf, None

    def evaluate_individual(self, chromo, data, target):
        """评估单个个体的适应度 - 改进的预测策略"""
        try:
            # 改进的预测策略：使用概率采样而非严格阈值
            should_evaluate = True
            
            if self.use_prediction and self.fitness_predictor.is_trained:
                predicted_fitness = self.fitness_predictor.predict(chromo)
                if predicted_fitness is not None:
                    # 改进：使用归一化的差距，避免负分数时的异常行为
                    if self.best_score > -np.inf and len(self.fitness_predictor.history['fitnesses']) > 1:
                        # 获取历史分数并过滤-inf
                        valid_scores = [s for s in self.fitness_predictor.history['fitnesses'] if s != -np.inf]
                        if len(valid_scores) > 1:
                            # 计算分数范围用于归一化
                            score_range = max(valid_scores) - min(valid_scores)
                            if score_range > 1e-10:
                                # 归一化差距到[0, 1]
                                normalized_gap = (self.best_score - predicted_fitness) / score_range
                                # 转换为评估概率：差距越大，评估概率越低
                                eval_probability = 1.0 - np.clip(normalized_gap, 0, 0.9)
                                eval_probability = np.clip(eval_probability, 0.1, 1.0)
                            else:
                                # 分数范围太小，全部评估
                                eval_probability = 1.0
                        else:
                            eval_probability = 1.0
                    else:
                        eval_probability = 1.0  # 初期全部评估
                    
                    should_evaluate = random.random() < eval_probability
                    
                    if not should_evaluate:
                        # 使用预测值（但标记为预测，不添加到历史）
                        return (predicted_fitness, chromo, None)
            
            # 评估实际适应度
            actual_fitness, model = self.fitness_function(chromo, data.copy(), target)
            
            # 过滤掉失败的评估
            if actual_fitness == -np.inf:
                logging.warning(f"个体评估失败，已过滤")
                return (-np.inf, chromo, None)
            
            self.fitness_predictor.add_history(chromo, actual_fitness)
            
            # 定期重新训练预测模型（改为每100个样本）
            if self.use_prediction:
                if len(self.fitness_predictor.history['chromosomes']) >= 50 and \
                   len(self.fitness_predictor.history['chromosomes']) % 100 == 0:
                    logging.info(f"重新训练预测模型... (已收集{len(self.fitness_predictor.history['chromosomes'])}个样本)")
                    self.fitness_predictor.train()
                    if self.fitness_predictor.is_trained:
                        logging.info("预测模型训练完成")
            
            # 记录超参数用于自适应范围调整
            if self.enable_adaptive_range and self.range_adjuster is not None:
                config = self.decode_chromosome(chromo)
                for param_name, param_value in config['hyperparameters'].items():
                    self.range_adjuster.record_evaluation(
                        config['model'],
                        param_name,
                        param_value,
                        actual_fitness
                    )

            return (actual_fitness, chromo, model)

        except Exception as e:
            logging.error(f"评估错误: {str(e)}")
            return (-np.inf, chromo, None)
    
    def adjust_hyperparameter_ranges(self, generation):
        """自动调整超参数搜索范围
        
        :param generation: 当前代数
        """
        if not self.enable_adaptive_range or self.range_adjuster is None:
            logging.debug(f"第{generation}代：超参数调整功能未启用")
            return
        
        if not self.range_adjuster.should_adjust(generation, adjust_interval=5):
            logging.debug(f"第{generation}代：不满足调整间隔条件，跳过调整")
            return
        
        logging.info(f"\n{'='*50}")
        logging.info(f"第{generation}代：开始自动调整超参数范围")
        logging.info(f"{'='*50}")
        
        adjusted_count = 0
        for model_name in list(self.model_hyperparameters.keys()):
            for param_name in list(self.model_hyperparameters[model_name].keys()):
                original_values = self.model_hyperparameters[model_name][param_name]
                
                # 计算调整后的范围
                new_values = self.range_adjuster.compute_optimal_range(
                    model_name, param_name, original_values
                )
                
                # 如果范围有变化，则更新
                if new_values != original_values:
                    self.model_hyperparameters[model_name][param_name] = new_values
                    logging.info(f"  ✓ {model_name}.{param_name}:")
                    logging.info(f"    原范围: {original_values}")
                    logging.info(f"    新范围: {new_values}")
                    adjusted_count += 1
        
        if adjusted_count > 0:
            logging.info(f"{'='*50}")
            logging.info(f"✓ 本次调整完成：共调整了 {adjusted_count} 个超参数范围")
            logging.info(f"✓ 累计调整次数：{self.range_adjuster.adjustment_count + 1}")
            logging.info(f"{'='*50}\n")
            self.range_adjuster.adjustment_count += 1
        else:
            logging.info(f"本次检查：所有超参数范围均未发生变化")
            logging.info(f"{'='*50}\n")
    
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
        
            # 过滤失败的个体
            valid_results = [(score, chromo, model) for score, chromo, model in results 
                            if score != -np.inf]
            
            if not valid_results:
                logging.warning(f"第{gen+1}代没有有效个体，跳过")
                continue
            
            scores = [s for s, _, _ in valid_results]
            valid_pop = [c for _, c, _ in valid_results]
            
            # 更新最佳个体
            for score, chromo, model in valid_results:
                if score > best_score:
                    best_score = score
                    best_config = chromo
                    best_model = model
                    # 保存最佳模型的特征列（去除目标变量）
                    self.best_model_features = [col for col in data.columns if col != target]
            
            self.best_score = best_score
            avg_fitness = np.mean(scores) if scores else -np.inf
            
            avg_history.append(avg_fitness)
            
            # 选择精英个体
            elite = sorted(zip(scores, valid_pop), key=lambda x: x[0], reverse=True)[:elite_size]
            elite = [c for (s, c) in elite]

            # 先调整超参数范围，再生成子代
            # 这样新种群可以使用调整后的范围
            if self.enable_adaptive_range and (gen + 1) % 5 == 0:
                self.adjust_hyperparameter_ranges(gen + 1)

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
                    model_name = models[model_idx] if model_idx < len(models) else models[0]
                    model_params = self.model_hyperparameters.get(model_name, {})
                    # 随机生成一个模型类型相同的新个体
                    p2 = p1.copy()
                    # 除了模型类型外，其他超参数基因随机生成
                    param_keys = list(model_params.keys())
                    for i in range(1, len(p2)):
                        param_idx = i - 1
                        if param_idx < len(param_keys):
                            param_values = model_params[param_keys[param_idx]]
                            if param_values:
                                p2[i] = random.randint(0, len(param_values) - 1)
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
            valid_count = len(valid_pop)
            # 只输出关键代际信息
            if (gen + 1) % 5 == 0 or gen == 0 or gen == generations - 1:
                logging.info(f"Gen {gen+1}/{generations} | Best: {history[-1]:.4f} | Avg: {avg_fitness:.4f} | Valid: {valid_count}/{population_size} | Time: {elapsed_time:.2f}s")
            else:
                logging.debug(f"Gen {gen+1}/{generations} | Best: {history[-1]:.4f} | Avg: {avg_fitness:.4f}")

        
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
            'feature_columns': processed_features if processed_features else [],  # 保存训练时的特征列
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
        #logging.info(f"模型已添加到集成候选列表，当前候选模型数量: {len(self.top_models)}")

    def integrated_predict(self, X, target=None):
        """
        对self.top_models中的模型进行加权平均集成 - 改进版本
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

            for idx, model_info in enumerate(self.top_models):
                model = model_info['model']
                model_name = model_info['model_name']
                try:
                    X_input = X.copy()
                    if self.target in X_input.columns:
                        X_input = X_input.drop(columns=[self.target])
                    
                    # 使用训练时保存的特征列，确保特征一致性
                    if 'feature_columns' in model_info and model_info['feature_columns']:
                        expected_features = model_info['feature_columns']
                        # 只选择存在且为数值类型的特征
                        available_features = []
                        for feat in expected_features:
                            if feat in X_input.columns and pd.api.types.is_numeric_dtype(X_input[feat]):
                                available_features.append(feat)
                        
                        if not available_features:
                            logging.warning(f"模型 {model_name} 没有可用的数值特征，跳过此模型")
                            continue
                        
                        X_input = X_input[available_features]
                    
                    else:
                        # fallback：使用数值特征
                        X_input = X_input.select_dtypes(include=[np.number])
                    
                    if X_input.empty:
                        logging.error(f"模型 {model_name} 没有可用的特征")
                        continue
                    
                    pred = model.predict(X_input)
                    predictions.append(pred)
                    
                    # 改进的权重策略：使用排名而非原始分数（避免量纲问题）
                    rank_weight = len(self.top_models) - idx
                    weights.append(rank_weight)
                    
                    logging.info(f"模型 {model_name} 预测完成，排名权重: {rank_weight}")
                except Exception as e:
                    logging.error(f"模型 {model_name} 预测失败: {str(e)}")
                    continue

            if not predictions:
                logging.error("所有模型预测都失败了")
                return None

            # 验证预测长度一致
            prediction_lengths = [len(pred) for pred in predictions]
            if len(set(prediction_lengths)) > 1:
                logging.error(f"预测结果长度不一致: {prediction_lengths}")
                return None

            predictions = np.array(predictions)
            weights = np.array(weights, dtype=float)

            # 归一化权重
            weights = weights / np.sum(weights)

            # 使用权重平均
            final_prediction = np.average(predictions, axis=0, weights=weights)
            final_prediction = np.array(final_prediction).flatten()

            # 分类任务的后处理
            if self.task_type in ['binary_classification', 'multiclass_classification']:
                final_prediction = np.round(final_prediction).astype(int)
                
                # 反向映射编码的标签
                if self.target_encoder is not None:
                    try:
                        final_prediction = self.target_encoder.inverse_transform(final_prediction)
                        logging.info("已对预测结果进行标签反向映射")
                    except Exception as e:
                        logging.warning(f"标签反向映射失败: {str(e)}")

            logging.info(f"集成预测完成，最终预测结果长度: {len(final_prediction)}")
            return final_prediction

        except Exception as e:
            logging.error(f"集成预测失败: {str(e)}")
            return None

if __name__ == "__main__":
    import load_data as ld
    from sklearn.model_selection import train_test_split
    data = ld.load_data("datasets/titanic_train.csv")
    target = "Fare"
    
    # 使用训练时保存的特征列，确保特征一致性
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)
    print(f"数据分割: 训练集={len(train_data)}行, 测试集={len(test_data)}行")
    
    # 1. 只在训练集上进行预处理优化
    Pre = pre.Preprocessing(data=train_data, target=target)
    Pre.run()
    processed_train = Pre.get_processed_data()
    print(f"训练集预处理完成: {processed_train.shape}")
    
    # 应用相同的预处理方案到测试集
    processed_test = Pre.transform(test_data)
    print(f"测试集预处理完成: {processed_test.shape}")
    
    # 2. 用处理后的训练集做模型/超参数搜索
    ga_ensemble = GeneticAlgorithm(
        data=processed_train,  # 使用训练集
        target=target,
        use_prediction=True, 
        enable_ensemble=True 
    )
    best_config, best_score, history, avg_history, best_model = ga_ensemble.run(
        generations=20,  
        population_size=10,  
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
            
            # 分离特征和目标变量（使用预处理后的测试集）
            if target in processed_test.columns:
                y_true = processed_test[target]
                X_test = processed_test.drop(columns=[target])
            
            # 使用训练时保存的特征列，确保特征一致性
            if ga_ensemble.best_model_features:
                expected_features = ga_ensemble.best_model_features
                # 只选择测试集中存在且为数值类型的特征
                available_features = []
                for feat in expected_features:
                    if feat in X_test.columns:
                        # 检查是否为数值类型
                        if pd.api.types.is_numeric_dtype(X_test[feat]):
                            available_features.append(feat)
                        else:
                            print(f"警告: 特征 '{feat}' 存在但不是数值类型，已跳过")
                    else:
                        print(f"警告: 特征 '{feat}' 在测试集中不存在，已跳过")
                
                if not available_features:
                    print(f"错误: 没有可用的数值特征")
                    raise ValueError("测试集中没有可用的训练特征")
                
                X_test = X_test[available_features]
                print(f"使用训练特征: {len(available_features)}/{len(expected_features)}个特征")
            else:
                # fallback：使用数值特征
                X_test = X_test.select_dtypes(include=[np.number])
                print(f"警告: 未找到保存的特征列，使用数值特征: {X_test.shape[1]}个")
            
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
    
    # 集成预测（使用预处理后的测试集）
    print("\n=== 集成模型测试 ===")
    result = ga_ensemble.integrated_predict(processed_test, target)
    if result is not None:
        y_pre, y_true = result, processed_test[target]
        
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


  
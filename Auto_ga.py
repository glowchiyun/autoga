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
import yaml

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
    def __init__(self, data, target, use_prediction=True, cv_scoring=None, enable_ensemble=False, enable_adaptive_range=True, task_type=None):
        """
        初始化遗传算法
        :param data: 输入数据
        :param target: 目标变量
        :param use_prediction: 是否使用预测模型
        :param cv_scoring: 交叉验证评分方法
        :param enable_ensemble: 是否启用集成
        :param enable_adaptive_range: 是否启用自适应超参数范围调整
        :param task_type: 任务类型 (可选，如果不传则自动检测)
                         'binary' -> 'binary_classification'
                         'multiclass' -> 'multiclass_classification'
                         'regression' -> 'regression'
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
        
        # 检测或使用指定的任务类型
        if task_type is not None:
            # 映射AMLB任务类型到GAML任务类型
            task_type_map = {
                'binary': 'binary_classification',
                'multiclass': 'multiclass_classification',
                'regression': 'regression'
            }
            self.task_type = task_type_map.get(task_type, task_type)
            # 对于分类任务，仍需检查是否需要编码器
            _, target_encoder = self.detect_task_type(data, target)
            logging.info(f"使用指定任务类型: {self.task_type} (原始: {task_type})")
        else:
            self.task_type, target_encoder = self.detect_task_type(data, target)
            logging.info(f"自动检测任务类型: {self.task_type}")  
        self.model_hyperparameters = self._get_default_model_hyperparameters()
        
        # 设置交叉验证评分方法 - 改进：使用更好的评分指标
        if cv_scoring is None:
            if self.task_type == 'regression':
                self.cv_scoring = 'neg_mean_squared_error'
            elif self.task_type == 'binary_classification':
                self.cv_scoring = 'roc_auc'  # 二分类使用AUC更合适
            else:  # multiclass_classification
                self.cv_scoring = 'accuracy'  # 多分类保持accuracy
        else:
            self.cv_scoring = cv_scoring
        self.target_encoder = target_encoder
        
        # 初始化其他属性
        self.best_score = -np.inf
        self.fitness_predictor = FitnessPredictor(model_hyperparameters=self.model_hyperparameters)
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
        """
        从配置文件加载模型超参数
        """
        config_path = os.path.join(os.path.dirname(__file__), 'model_hyperparameters.yaml')
        
        # 尝试从 YAML 文件加载
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 将 YAML 中的 null 转换为 None，列表转换为元组（用于 hidden_layer_sizes）
                def convert_values(params):
                    converted = {}
                    for key, values in params.items():
                        if isinstance(values, list):
                            converted_list = []
                            for v in values:
                                if v is None:
                                    converted_list.append(None)
                                elif isinstance(v, list):
                                    converted_list.append(tuple(v))  # 转换嵌套列表为元组
                                else:
                                    converted_list.append(v)
                            converted[key] = converted_list
                        else:
                            converted[key] = values
                    return converted
                
                # 根据任务类型返回配置
                if self.task_type == 'regression':
                    params = config.get('regression', {})
                elif self.task_type == 'binary_classification':
                    params = config.get('binary_classification', {})
                else:  # multiclass_classification
                    params = config.get('multiclass_classification', {})
                
                # 转换所有模型的参数
                result = {}
                for model_name, model_params in params.items():
                    result[model_name] = convert_values(model_params)
                
                logging.info(f"从配置文件加载超参数: {config_path}")
                return result
                
            except Exception as e:
                error_msg = f"加载配置文件失败: {e}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            error_msg = f"配置文件不存在: {config_path}，请确保 model_hyperparameters.yaml 文件存在"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

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
                score, model, actual_features = rg.Regressor(
                    dataset=processed_data,
                    target=target,
                    strategy=config['model'],
                    hyperparameters=config['hyperparameters'],
                    cv_scoring=self.cv_scoring
                ).transform()

                if self.enable_ensemble and model is not None and score > -np.inf:
                    # 使用模型实际使用的特征列（由Regressor返回）
                    self.add_top_models(chromosome, score, model, processed_features=actual_features)

                return score, model
            else:  # classification tasks
                score, model, actual_features = cl.Classifier(
                    dataset=processed_data,
                    target=target,
                    strategy=config['model'],
                    hyperparameters=config['hyperparameters'],
                    cv_scoring=self.cv_scoring
                ).transform()

                if self.enable_ensemble and model is not None and score > -np.inf:
                    # 使用模型实际使用的特征列（由Classifier返回）
                    self.add_top_models(chromosome, score, model, processed_features=actual_features)
                return score, model
        except Exception as e:
            logging.error(f"Fitness calculation error: {str(e)}")
            return -np.inf, None

    def evaluate_individual(self, chromo, data, target):
        """评估单个个体的适应度 - 改进的预测策略（使用置信度调节）"""
        try:
            # 改进的预测策略：结合不确定性估计决定是否跳过评估
            should_evaluate = True
            
            if self.use_prediction and self.fitness_predictor.is_trained:
                confidence = self.fitness_predictor.get_confidence()
                
                if confidence > 0.3:  # 降低门槛到0.3（使代理模型更早发挥作用）
                    # 使用不确定性估计（RF树方差）进行更智能的跳过决策
                    predicted_fitness, uncertainty = self.fitness_predictor.predict_with_uncertainty(chromo)
                    
                    if predicted_fitness is not None and self.best_score > -np.inf:
                        # 计算相对差距
                        relative_gap = (self.best_score - predicted_fitness) / max(abs(self.best_score), 0.01)
                        
                        # 策略：预测分数远低于最佳分数，且不确定性低时，跳过评估
                        # 不确定性高意味着模型"不确定"，应该评估以获取信息
                        uncertainty_ratio = uncertainty / max(abs(predicted_fitness), 0.01) if uncertainty is not None else 1.0
                        
                        if relative_gap > 0.05 and uncertainty_ratio < 0.3:
                            # 跳过概率随差距增大和置信度增加而增大
                            base_skip = min(0.6, relative_gap * 4)
                            skip_probability = base_skip * min(confidence, 1.0)
                            should_evaluate = random.random() > skip_probability
                            
                            if not should_evaluate:
                                logging.debug(f"跳过评估: 预测={predicted_fitness:.4f}, 最佳={self.best_score:.4f}, "
                                            f"差距={relative_gap:.2%}, 不确定性={uncertainty:.4f}, 置信度={confidence:.2f}")
                                return (predicted_fitness, chromo, None)
                    # 如果预测分数接近或高于最佳分数，或不确定性高，仍然全部评估
            
            # 评估实际适应度
            actual_fitness, model = self.fitness_function(chromo, data.copy(), target)
            
            # 过滤掉失败的评估
            if actual_fitness == -np.inf:
                logging.warning(f"个体评估失败，已过滤")
                return (-np.inf, chromo, None)
            
            self.fitness_predictor.add_history(chromo, actual_fitness)
            
            # 智能重训练：使用 FitnessPredictor 自身的判断
            if self.use_prediction and self.fitness_predictor.should_retrain():
                n_samples = len(self.fitness_predictor.history['chromosomes'])
                logging.info(f"重新训练预测模型... (已收集{n_samples}个样本)")
                success = self.fitness_predictor.train()
                if success:
                    confidence = self.fitness_predictor.get_confidence()
                    logging.info(f"✓ 预测模型训练完成 (R²={confidence:.4f})")
                else:
                    logging.warning("✗ 预测模型训练失败")
            
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
        
        # 检查是否已达到最大调整次数
        if self.range_adjuster.adjustment_count >= self.range_adjuster.max_adjustments:
            logging.info(f"第{generation}代：已达到最大调整次数({self.range_adjuster.max_adjustments})，停止调整以保持搜索多样性")
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
            
            # 同步更新适应度预测器的模型超参数信息（范围变化后特征工程需要更新）
            if self.use_prediction and hasattr(self.fitness_predictor, 'update_model_hyperparameters'):
                self.fitness_predictor.update_model_hyperparameters(self.model_hyperparameters)
                logging.info("✓ 已同步更新适应度预测器的超参数范围信息")
        else:
            logging.info(f"本次检查：所有超参数范围均未发生变化")
            logging.info(f"{'='*50}\n")
    
    def run(self, generations=20, population_size=50, mutation_rate=0.2, 
            elite_size=2, n_jobs=-1, time_limit=None, tournament_size=3,
            early_stop_patience=8, min_improvement=1e-6, adaptive_mutation=True,
            mode='generations'):
        """
        运行遗传算法 - 支持两种运行模式
        
        :param generations: 迭代次数（mode='generations'时有效）
        :param population_size: 种群大小
        :param mutation_rate: 变异率
        :param elite_size: 精英个体数量
        :param n_jobs: 并行数
        :param time_limit: 时间限制（秒），mode='time_budget'时必须指定
        :param tournament_size: 锦标赛规模
        :param early_stop_patience: 早停耐心值，连续多少代无改进则停止（仅mode='generations'时有效）
        :param min_improvement: 最小改进阈值，低于此值视为无改进
        :param adaptive_mutation: 是否启用自适应变异率
        :param mode: 运行模式
                    - 'generations': 基于代数，进化到指定代数停止（默认）
                    - 'time_budget': 基于时间预算，不断优化直到时间耗尽
        :return: (best_config, best_score, history, avg_history, best_model)
        """
        # 参数验证
        if mode == 'time_budget' and time_limit is None:
            raise ValueError("time_budget 模式必须指定 time_limit 参数")
        
        if mode == 'time_budget':
            logging.info(f"=== 启动时间预算模式 ===")
            logging.info(f"时间限制: {time_limit}秒 ({time_limit/60:.1f}分钟)")
            logging.info(f"策略: 持续优化直到时间耗尽，充分利用计算资源")
            return self._run_time_budget_mode(
                population_size=population_size,
                mutation_rate=mutation_rate,
                elite_size=elite_size,
                n_jobs=n_jobs,
                time_limit=time_limit,
                tournament_size=tournament_size,
                min_improvement=min_improvement,
                adaptive_mutation=adaptive_mutation
            )
        else:
            logging.info(f"=== 启动代数模式 ===")
            logging.info(f"进化代数: {generations}")
            logging.info(f"早停策略: patience={early_stop_patience}, min_improvement={min_improvement}")
            return self._run_generations_mode(
                generations=generations,
                population_size=population_size,
                mutation_rate=mutation_rate,
                elite_size=elite_size,
                n_jobs=n_jobs,
                time_limit=time_limit,
                tournament_size=tournament_size,
                early_stop_patience=early_stop_patience,
                min_improvement=min_improvement,
                adaptive_mutation=adaptive_mutation
            )
    
    def _run_generations_mode(self, generations, population_size, mutation_rate, 
                             elite_size, n_jobs, time_limit, tournament_size,
                             early_stop_patience, min_improvement, adaptive_mutation):
        """
        基于代数的运行模式（原有逻辑）
        """
        data=self.data
        target=self.target
        population = self.initialize_population(population_size)
        best_config = None
        best_score = -np.inf
        best_model = None
        history = []
        avg_history = []
        
        # 早停相关变量
        no_improvement_count = 0
        previous_best = -np.inf
        current_mutation_rate = mutation_rate
        
        start_time = time.time()
        
        for gen in range(generations):
            # 检查时间限制
            if time_limit is not None and time.time() - start_time > time_limit:
                logging.info(f"达到时间限制 {time_limit} 秒，提前结束进化")
                break
            
            # 早停检查
            if early_stop_patience is not None and no_improvement_count >= early_stop_patience:
                logging.info(f"连续 {early_stop_patience} 代无改进，触发早停机制")
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
            
            # 早停检测：检查是否有显著改进
            improvement = best_score - previous_best
            if improvement > min_improvement:
                no_improvement_count = 0
                previous_best = best_score
            else:
                no_improvement_count += 1
            
            # 自适应变异率：无改进时增加变异率以跳出局部最优
            if adaptive_mutation:
                if no_improvement_count >= 3:
                    # 逐步增加变异率，但不超过0.5
                    current_mutation_rate = min(0.5, mutation_rate * (1 + 0.15 * no_improvement_count))
                    if no_improvement_count == 3:
                        logging.info(f"启用自适应变异率: {current_mutation_rate:.3f}")
                else:
                    current_mutation_rate = mutation_rate
            
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
                    next_gen.append(self.mutate(c1, current_mutation_rate))
                    next_gen.append(self.mutate(c2, current_mutation_rate))
            else:
                while len(next_gen) < population_size - elite_size:
                    # 随机选一个有两个及以上个体的模型
                    model_idx = random.choice(candidate_model_indices)
                    idxs = model_to_indices[model_idx]
                    # 随机选两个不同的个体
                    # 确保有足够的索引
                    if len(idxs) < 2:
                        continue
                    p1_idx, p2_idx = random.sample(idxs, 2)
                    p1 = selected_population[p1_idx]
                    p2 = selected_population[p2_idx]
                    c1, c2 = self.crossover(p1, p2)
                    next_gen.append(self.mutate(c1, current_mutation_rate))
                    next_gen.append(self.mutate(c2, current_mutation_rate))
            
            population = next_gen[:population_size - elite_size] + elite
            
            # 直接使用原始适应度值记录历史
            history.append(best_score)
                
            elapsed_time = time.time() - start_time
            # 统计实际评估的个体数量（model不为None表示实际评估）
            actual_evaluated = sum(1 for _, _, model in valid_results if model is not None)
            predicted_count = len(valid_results) - actual_evaluated
            valid_count = len(valid_pop)
            # 只输出关键代际信息
            if (gen + 1) % 5 == 0 or gen == 0 or gen == generations - 1:
                stagnation_info = f" | Stagnation: {no_improvement_count}" if no_improvement_count > 0 else ""
                mutation_info = f" | MutRate: {current_mutation_rate:.3f}" if current_mutation_rate != mutation_rate else ""
                if predicted_count > 0:
                    logging.info(f"Gen {gen+1}/{generations} | Best: {history[-1]:.4f} | Avg: {avg_fitness:.4f} | Eval: {actual_evaluated}/{population_size} (预测跳过: {predicted_count}){stagnation_info}{mutation_info} | Time: {elapsed_time:.2f}s")
                else:
                    logging.info(f"Gen {gen+1}/{generations} | Best: {history[-1]:.4f} | Avg: {avg_fitness:.4f} | Valid: {valid_count}/{population_size}{stagnation_info}{mutation_info} | Time: {elapsed_time:.2f}s")
            else:
                logging.debug(f"Gen {gen+1}/{generations} | Best: {history[-1]:.4f} | Avg: {avg_fitness:.4f}")

        
        return best_config, best_score, history, avg_history, best_model
    
    def _run_time_budget_mode(self, population_size, mutation_rate, elite_size, 
                             n_jobs, time_limit, tournament_size, min_improvement, 
                             adaptive_mutation):
        """
        基于时间预算的运行模式 - 持续优化直到时间耗尽
        
        策略：
        1. 设置时间缓冲（预留5%时间用于最后的集成预测和收尾）
        2. 持续进化，每代结束后检查时间
        3. 时间耗尽时立即停止，返回当前最优模型
        4. 保持全局最优模型和top_models列表
        """
        data = self.data
        target = self.target
        
        # 时间管理
        start_time = time.time()
        time_buffer = time_limit * 0.05  # 预留5%时间缓冲
        effective_time_limit = time_limit - time_buffer
        
        # 全局最优跟踪
        global_best_config = None
        global_best_score = -np.inf
        global_best_model = None
        global_history = []
        global_avg_history = []
        
        # 初始化种群（只需一次）
        population = self.initialize_population(population_size)
        
        # 进化参数
        generation = 0
        current_mutation_rate = mutation_rate
        no_improvement_count = 0  # 连续无改进代数
        previous_best = -np.inf
        
        logging.info(f"时间预算模式启动:")
        logging.info(f"  总时间预算: {time_limit}秒")
        logging.info(f"  有效优化时间: {effective_time_limit}秒")
        logging.info(f"  时间缓冲: {time_buffer}秒")
        
        while True:
            # 检查时间是否耗尽
            elapsed = time.time() - start_time
            remaining = effective_time_limit - elapsed
            
            if remaining <= 0:
                logging.info(f"\n{'='*60}")
                logging.info(f"时间预算耗尽，停止优化")
                logging.info(f"总耗时: {elapsed:.1f}秒 / {time_limit}秒")
                logging.info(f"完成代数: {generation}")
                logging.info(f"{'='*60}")
                break
            
            generation += 1
            
            # 并行评估种群适应度
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self.evaluate_individual)(chromo, data, target) 
                for chromo in population
            )
            
            # 过滤失败的个体
            valid_results = [(score, chromo, model) for score, chromo, model in results 
                            if score != -np.inf]
            
            if not valid_results:
                logging.warning(f"第{generation}代: 没有有效个体")
                continue
            
            scores = [s for s, _, _ in valid_results]
            valid_pop = [c for _, c, _ in valid_results]
            
            # 更新全局最佳 - 关键修复：只有实际评估过的个体（model不为None）才能更新全局最优
            # 适应度预测跳过的个体返回 (predicted_fitness, chromo, None)，
            # 其预测值可能因误差而虚高，不应覆盖真正训练过的最优模型
            for score, chromo, model in valid_results:
                if score > global_best_score and model is not None:
                    global_best_score = score
                    global_best_config = chromo
                    global_best_model = model
                    self.best_score = global_best_score
                    logging.info(f"  ✓ 新全局最优! 第{generation}代 | Score: {global_best_score:.4f} | 剩余: {remaining:.1f}s")
            
            avg_fitness = np.mean(scores)
            global_history.append(global_best_score)
            global_avg_history.append(avg_fitness)
            
            # 早停检测：检查是否有显著改进
            improvement = global_best_score - previous_best
            if improvement > min_improvement:
                no_improvement_count = 0
                previous_best = global_best_score
            else:
                no_improvement_count += 1
            
            # 自适应变异率
            if adaptive_mutation:
                if no_improvement_count > 0:
                    current_mutation_rate = min(mutation_rate * 1.5, 0.5)
                else:
                    current_mutation_rate = mutation_rate
            
            # 选择精英
            elite = sorted(zip(scores, valid_pop), key=lambda x: x[0], reverse=True)[:elite_size]
            elite = [c for (s, c) in elite]
            
            # 超参数范围自适应调整
            if self.enable_adaptive_range and generation % 5 == 0:
                self.adjust_hyperparameter_ranges(generation)
            
            # 锦标赛选择
            selected_population = self.tournament_selection(valid_pop, scores, tournament_size)
            
            # 交叉变异生成子代
            next_gen = []
            from collections import defaultdict
            model_to_indices = defaultdict(list)
            models = list(self.model_hyperparameters.keys())
            
            for idx, chromo in enumerate(selected_population):
                if len(chromo) > 0:
                    model_idx = chromo[0]
                    if model_idx < len(models):
                        model_to_indices[model_idx].append(idx)
            
            candidate_model_indices = [m for m, idxs in model_to_indices.items() if len(idxs) >= 2]
            
            if not candidate_model_indices:
                while len(next_gen) < population_size - elite_size:
                    idx = random.randint(0, len(selected_population) - 1)
                    parent1 = selected_population[idx]
                    if len(parent1) > 0:
                        model_idx = parent1[0]
                        if model_idx < len(models):
                            new_chromo = [model_idx]
                            model_name = models[model_idx]
                            hyper_params = self.model_hyperparameters[model_name]
                            for param_name, values in hyper_params.items():
                                if len(values) > 0:
                                    new_chromo.append(random.randint(0, len(values) - 1))
                                else:
                                    new_chromo.append(0)
                            max_hyper = max(len(p) for p in self.model_hyperparameters.values())
                            remaining_params = max_hyper - len(hyper_params)
                            new_chromo += [0] * remaining_params
                            parent2 = new_chromo
                        else:
                            parent2 = selected_population[random.randint(0, len(selected_population) - 1)]
                    else:
                        parent2 = selected_population[random.randint(0, len(selected_population) - 1)]
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    next_gen.append(self.mutate(child1, current_mutation_rate))
                    if len(next_gen) < population_size - elite_size:
                        next_gen.append(self.mutate(child2, current_mutation_rate))
            else:
                while len(next_gen) < population_size - elite_size:
                    chosen_model_idx = random.choice(candidate_model_indices)
                    candidates = [selected_population[i] for i in model_to_indices[chosen_model_idx]]
                    parent1, parent2 = random.sample(candidates, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    next_gen.append(self.mutate(child1, current_mutation_rate))
                    if len(next_gen) < population_size - elite_size:
                        next_gen.append(self.mutate(child2, current_mutation_rate))
            
            population = next_gen[:population_size - elite_size] + elite
            
            # 每5代输出一次进度信息
            if generation % 5 == 0:
                stagnation_info = f" | 停滞: {no_improvement_count}代" if no_improvement_count > 0 else ""
                mutation_info = f" | 变异率: {current_mutation_rate:.3f}" if current_mutation_rate != mutation_rate else ""
                logging.info(f"第{generation}代 | 最优: {global_best_score:.4f} | 平均: {avg_fitness:.4f} | 已用: {elapsed:.1f}s | 剩余: {remaining:.1f}s{stagnation_info}{mutation_info}")
        
        # 最终统计
        total_elapsed = time.time() - start_time
        logging.info(f"\n{'='*60}")
        logging.info(f"时间预算模式完成")
        logging.info(f"{'='*60}")
        logging.info(f"总耗时: {total_elapsed:.1f}秒 / {time_limit}秒 (利用率: {total_elapsed/time_limit*100:.1f}%)")
        logging.info(f"完成代数: {generation}")
        logging.info(f"最终得分: {global_best_score:.4f}")
        logging.info(f"Top模型数量: {len(self.top_models)}")
        logging.info(f"{'='*60}\n")
        
        return global_best_config, global_best_score, global_history, global_avg_history, global_best_model

    def add_top_models(self, chromosome, score, model, processed_features=None):
        """
        维护ensemble_size大小的top_models，保证其中模型类型各不相同且分数最优。
        :param chromosome: 染色体（配置）
        :param score: 模型得分
        :param model: 训练好的模型
        :param processed_features: 预处理后的特征列名列表
        """
        # 回归任务质量门控：拒绝与最佳模型差距过大的模型进入候选池
        if self.task_type == 'regression' and len(self.top_models) > 0:
            best_existing_score = self.top_models[0]['score']  # 已按分数降序排列
            if abs(best_existing_score) > 1e-8:
                # 对于neg_MSE（负值，越接近0越好），只保留MSE不超过最佳3倍的模型
                # 例: best=-100, threshold=-300, score=-500 会被拒绝
                quality_threshold = best_existing_score * 3
                if score < quality_threshold:
                    logging.debug(f"回归质量门控: 拒绝低质量模型 (score={score:.4f}, "
                                  f"best={best_existing_score:.4f}, threshold={quality_threshold:.4f})")
                    return
        
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

    def integrated_predict(self, X, target=None, voting='soft'):
        """
        对self.top_models中的模型进行集成预测 - 高级版本
        
        改进点：
        1. 分类任务使用概率预测+加权投票（软投票/硬投票）
        2. 更智能的权重策略（指数权重+平方根平滑）
        3. 异常预测检测和处理
        4. 回归任务使用加权中位数（对异常值更鲁棒）
        
        :param X: 输入数据
        :param target: 目标变量名
        :param voting: 投票方式，'soft'(软投票，基于概率) 或 'hard'(硬投票，基于类别)
        :return: 集成预测结果
        """
        if len(self.top_models) == 0:
            logging.error("集成模型未训练或没有模型")
            return None

        try:
            predictions = []
            probabilities = []  # 用于分类任务的概率预测
            scores = []
            valid_models = []

            for idx, model_info in enumerate(self.top_models):
                model = model_info['model']
                model_name = model_info['model_name']
                model_score = model_info['score']
                try:
                    X_input = X.copy()
                    if self.target in X_input.columns:
                        X_input = X_input.drop(columns=[self.target])
                    
                    # 使用训练时保存的特征列，确保特征一致性
                    if 'feature_columns' in model_info and model_info['feature_columns']:
                        expected_features = model_info['feature_columns']
                        
                        # 步骤1：删除训练时没有的额外特征
                        extra_features = [f for f in X_input.columns if f not in expected_features]
                        if extra_features:
                            logging.warning(f"模型 {model_name} 测试集有训练时不存在的特征，将删除: {extra_features}")
                            X_input = X_input.drop(columns=extra_features)
                        
                        # 步骤2：添加训练时有但测试集缺失的特征（填充0）
                        for feat in expected_features:
                            if feat not in X_input.columns:
                                logging.warning(f"模型 {model_name} 缺少特征 '{feat}'，填充为0")
                                X_input[feat] = 0
                            elif not pd.api.types.is_numeric_dtype(X_input[feat]):
                                logging.warning(f"模型 {model_name} 特征 '{feat}' 不是数值类型，转换为数值")
                                try:
                                    X_input[feat] = pd.to_numeric(X_input[feat], errors='coerce').fillna(0)
                                except:
                                    X_input[feat] = 0
                        
                        # 步骤3：按训练时的特征顺序选择
                        X_input = X_input[expected_features]
                        logging.info(f"模型 {model_name} 使用 {len(expected_features)} 个特征进行预测")
                    else:
                        # fallback：使用数值特征
                        X_input = X_input.select_dtypes(include=[np.number])
                    
                    if X_input.empty:
                        logging.error(f"模型 {model_name} 没有可用的特征")
                        continue
                    
                    # 分类任务：获取概率预测和类别预测
                    if self.task_type in ['binary_classification', 'multiclass_classification']:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_input)
                            probabilities.append(proba)
                            pred = model.predict(X_input)
                            # Diagnostic: log model.classes_ and sample probabilities for debugging
                            if idx < 3:  # Only log first 3 models to avoid spam
                                model_classes = getattr(model, 'classes_', 'N/A')
                                logging.info(f"[DIAG] Model {idx} ({model_name}): classes_={model_classes}, "
                                           f"proba shape={proba.shape}, "
                                           f"proba[0]={proba[0] if len(proba) > 0 else 'empty'}, "
                                           f"pred[0]={pred[0] if len(pred) > 0 else 'empty'}")
                        else:
                            # 模型不支持概率预测，只能使用硬投票
                            pred = model.predict(X_input)
                            probabilities.append(None)
                    else:
                        # 回归任务
                        pred = model.predict(X_input)
                    
                    # 修复：统一展平预测结果为1维数组（CatBoost返回2维需要ravel）
                    pred = np.asarray(pred).ravel()
                    predictions.append(pred)
                    scores.append(model_score)
                    valid_models.append(model_name)
                    
                    logging.info(f"模型 {model_name} 预测完成，得分: {model_score:.4f}")
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
            scores = np.array(scores, dtype=float)
            
            # 改进的权重策略：指数权重 + 平方根平滑
            if np.min(scores) == np.max(scores):
                weights = np.ones(len(scores))
            else:
                # 归一化分数到 [0, 1]
                normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
                # 使用指数权重，但用平方根平滑，避免过度偏向最优模型
                weights = np.exp(3 * normalized_scores)  # 提高指数系数到3
                # 平方根平滑，让权重分布更均匀
                weights = np.sqrt(weights)
            
            # 归一化权重
            weights = weights / np.sum(weights)
            
            logging.info(f"集成权重: {dict(zip(valid_models, weights.round(3)))}")

            # 根据任务类型选择集成策略
            if self.task_type in ['binary_classification', 'multiclass_classification']:
                # 分类任务：使用投票机制
                if voting == 'soft' and all(p is not None for p in probabilities):
                    # 软投票：基于概率的加权平均
                    probabilities = np.array(probabilities)  # shape: (n_models, n_samples, n_classes)
                    weighted_proba = np.average(probabilities, axis=0, weights=weights)
                    final_prediction = np.argmax(weighted_proba, axis=1)
                    ensemble_proba = weighted_proba  # 保存加权概率用于返回
                    logging.info("使用软投票（概率加权平均）")
                else:
                    # 硬投票：基于类别的加权投票
                    ensemble_proba = None
                    n_samples = len(predictions[0])
                    final_prediction = np.zeros(n_samples, dtype=int)
                    
                    for i in range(n_samples):
                        votes = [pred[i] for pred in predictions]
                        vote_counts = {}
                        for vote, weight in zip(votes, weights):
                            vote_counts[vote] = vote_counts.get(vote, 0) + weight
                        final_prediction[i] = max(vote_counts.items(), key=lambda x: x[1])[0]
                    
                    logging.info("使用硬投票（类别加权投票）")
                
                # 注意：不再做 inverse_transform，让调用方(exec.py)统一处理标签转换
                # exec.py 的 save_predictions 会根据 target_is_encoded=True 做反向映射
            
            else:
                # 回归任务：质量门控集成预测
                # 核心改进：在模型级别过滤质量差距过大的模型，而非逐样本检测异常
                # 原因：回归任务中，低质量模型的预测会系统性偏移，逐样本过滤无法解决此问题
                
                best_score_val = np.max(scores)  # 最佳模型得分（neg_MSE，越接近0越好）
                
                # 质量门控：只保留得分在最佳模型合理范围内的模型
                # 对于neg_MSE: best=-100, threshold=-200, 即只保留MSE不超过最佳2倍的模型
                if abs(best_score_val) > 1e-8:
                    quality_threshold = best_score_val * 2  # 允许2倍MSE降级
                    quality_mask = scores >= quality_threshold
                else:
                    quality_mask = np.ones(len(scores), dtype=bool)
                
                n_quality_models = int(quality_mask.sum())
                quality_model_names = [valid_models[i] for i in range(len(valid_models)) if quality_mask[i]]
                
                logging.info(f"回归质量门控: {n_quality_models}/{len(scores)}个模型通过 "
                            f"(阈值={quality_threshold if abs(best_score_val) > 1e-8 else 'N/A'}, "
                            f"最佳={best_score_val:.4f})")
                logging.info(f"通过模型: {quality_model_names}")
                logging.info(f"所有模型得分: {dict(zip(valid_models, scores.round(4)))}")
                
                if n_quality_models <= 1:
                    # 只有最佳模型通过质量门控，直接使用最佳单模型预测
                    final_prediction = predictions[0]  # top_models按分数降序排列，index 0为最佳
                    logging.info(f"回归集成: 仅最佳模型通过质量门控，使用单模型预测")
                else:
                    # 使用通过质量门控的模型进行加权集成
                    filtered_predictions = predictions[quality_mask]
                    filtered_scores = scores[quality_mask]
                    
                    # 更激进的权重策略：高指数系数 + 无平方根平滑
                    # 让最佳模型拥有压倒性权重，避免次优模型拉低整体表现
                    if np.min(filtered_scores) == np.max(filtered_scores):
                        filtered_weights = np.ones(len(filtered_scores))
                    else:
                        norm_scores = (filtered_scores - np.min(filtered_scores)) / \
                                     (np.max(filtered_scores) - np.min(filtered_scores) + 1e-8)
                        filtered_weights = np.exp(5 * norm_scores)  # 指数系数5（比分类任务的3更激进）
                        # 注意：不使用sqrt平滑，让最佳模型权重更高
                    
                    filtered_weights = filtered_weights / filtered_weights.sum()
                    
                    logging.info(f"回归集成权重: {dict(zip(quality_model_names, filtered_weights.round(4)))}")
                    
                    # 向量化加权平均（高效且无需逐样本循环）
                    final_prediction = np.average(filtered_predictions, axis=0, weights=filtered_weights)
                
                logging.info("使用质量门控加权平均（回归专用）")

            final_prediction = np.array(final_prediction).flatten()
            logging.info(f"集成预测完成，最终预测结果长度: {len(final_prediction)}")
            
            # 分类任务返回 (predictions, probabilities)，回归任务返回 (predictions, None)
            if self.task_type in ['binary_classification', 'multiclass_classification']:
                if ensemble_proba is not None:
                    logging.info(f"返回集成预测和加权概率，概率shape: {ensemble_proba.shape}")
                return (final_prediction, ensemble_proba)
            else:
                return (final_prediction, None)

        except Exception as e:
            logging.error(f"集成预测失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
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


  
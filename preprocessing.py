import logging
import time
import pandas as pd
import normalizer as nl
import imputer as imp
import outlier_detector as out
import duplicate_detector as dup
import feature_selector as fs
import regressor as rg
import classifier as cl
import encoder as enc
import random
import numpy as np
from sklearn.model_selection import cross_val_score

class Preprocessing:
    def __init__(self, data, target, generations=10, population_size=10, mutation_rate=0.2, elite_size=2):
        self.data = data
        self.target = target
        self.task_type,target_encoder = self.detect_task_type(data, target)
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        enc.global_encoder.train_on_data(data, ratio_threshold=0.5, count_threshold=8)
        self.preprocessing_steps = self._get_default_preprocessing_steps()
        self.pre_len = len(self.preprocessing_steps)
        self.best_plan = None
        self.best_score = -float('inf')
        self.history = []

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

    def initialize_population(self):
        steps = self.preprocessing_steps
        population = []
        for _ in range(self.population_size):
            chromo = []
            for strategies in steps.values():
                if len(strategies) > 0:
                    chromo.append(random.randint(0, len(strategies)-1))
                else:
                    chromo.append(0)
            population.append(chromo)
        return population

    def decode_chromosome(self, chromosome):
        decoded = {}
        index = 0
        for step, strategies in self.preprocessing_steps.items():
            if index < len(chromosome) and len(strategies) > 0:
                strategy_idx = min(chromosome[index], len(strategies)-1)
                decoded[step] = strategies[strategy_idx]
            else:
                decoded[step] = strategies[0] if strategies else ''
            index += 1
        return decoded

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1)-1)
        return (
            parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:],
        )

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                strategies = list(self.preprocessing_steps.values())[i]
                if len(strategies) > 0:
                    chromosome[i] = random.randint(0, len(strategies)-1)
        return chromosome

    def tournament_selection(self, population, fitnesses, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitnesses[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected

    def fitness_function(self, chromo):
        plan = self.decode_chromosome(chromo)
        try:
            X = self.data.copy()
            processed_data = self.execute_preprocessing_plan(X, self.target, plan)
            if self.task_type == 'regression':
                score, _ = rg.Regressor(
                    dataset=processed_data,
                    target=self.target,
                ).transform()
            elif self.task_type == 'binary_classification':
                score, _ = cl.Classifier(
                    dataset=processed_data,
                    target=self.target,
                ).transform()
            else:
                score, _ = cl.Classifier(
                    dataset=processed_data,
                    target=self.target,
                ).transform()
            return score
        except Exception as e:
            return -float('inf')

    def run(self):
        population = self.initialize_population()
        best_plan = None
        best_score = -float('inf')
        for gen in range(self.generations):
            fitnesses = [self.fitness_function(chromo) for chromo in population]
            self.history.append(max(fitnesses))
            elite = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)[:self.elite_size]
            elite = [c for (s, c) in elite]
            selected = self.tournament_selection(population, fitnesses)
            next_gen = []
            while len(next_gen) < self.population_size - self.elite_size:
                p1, p2 = random.sample(selected, 2)
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutate(c1))
                next_gen.append(self.mutate(c2))
            population = next_gen[:self.population_size - self.elite_size] + elite
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_score:
                best_score = fitnesses[max_idx]
                best_plan = population[max_idx]
        self.best_plan = self.decode_chromosome(best_plan)
        self.best_score = best_score
        return self.best_plan, self.best_score

    def get_processed_data(self):
        """获取使用最佳预处理方案处理后的训练数据"""
        if self.best_plan is None:
            raise ValueError("请先运行run()方法找到最佳预处理方案")
        Xy = self.data.copy()
        Xy = self.execute_preprocessing_plan(Xy, self.target, self.best_plan)
        return Xy
    
    def transform(self, new_data):
        """
        使用已找到的最佳预处理方案处理新数据（如测试集）
        
        Parameters:
        -----------
        new_data : pd.DataFrame
            待处理的新数据（例如测试集）
            
        Returns:
        --------
        pd.DataFrame
            使用相同预处理方案处理后的数据
            
        Notes:
        ------
        - 必须先调用run()方法找到最佳预处理方案
        - 使用训练时保存的编码器状态确保一致性
        - 确保目标变量存在于新数据中
        """
        if self.best_plan is None:
            raise ValueError("请先运行run()方法找到最佳预处理方案")
        
        if self.target not in new_data.columns:
            raise ValueError(f"目标变量 {self.target} 不在新数据中")
        
        # 使用相同的预处理方案处理新数据
        # 注意：编码器会使用全局编码器的训练状态
        transformed_data = self.execute_preprocessing_plan(
            new_data.copy(), 
            self.target, 
            self.best_plan
        )
        
        logging.info(f"已使用最佳预处理方案转换新数据: {len(new_data)}行 -> {len(transformed_data)}行")
        return transformed_data
    
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
                    X = imp.Imputer(X, strategy=preprocessing_plan['imputer'], verbose=False).transform()
                    preprocessing_log.append(f"缺失值处理: {preprocessing_plan['imputer']}")
                except Exception as e:
                    logging.error(f"缺失值处理失败: {str(e)}")
                    X = X.ffill().bfill()  # 使用新的API替代废弃的fillna(method=)
                    preprocessing_log.append("缺失值处理: 使用前向填充")
            
            # 2. 异常值检测和处理
            if 'outliers' in preprocessing_plan and preprocessing_plan['outliers']:
                try:
                    X = out.Outlier_detector(X, strategy=preprocessing_plan['outliers'], threshold=0.8, verbose=False).transform()
                    preprocessing_log.append(f"异常值处理: {preprocessing_plan['outliers']}")
                except Exception as e:
                    logging.error(f"异常值检测失败: {str(e)}")
                    preprocessing_log.append("异常值处理: 跳过")
            
            # 3. 重复值检测和处理
            if 'duplicate_detector' in preprocessing_plan and preprocessing_plan['duplicate_detector']:
                try:
                    original_size = len(X)
                    X = dup.Duplicate_detector(X, strategy=preprocessing_plan['duplicate_detector'], verbose=False).transform()
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
                X = enc.Encoder(dataset=X, ratio_threshold=0.5, count_threshold=50).transform()
                preprocessing_log.append("特征编码: 自动编码")
            except Exception as e:
                logging.error(f"特征编码失败: {str(e)}")
                preprocessing_log.append("特征编码: 跳过")
            
            # 5. 数据标准化/归一化
            if 'normalizer' in preprocessing_plan and preprocessing_plan['normalizer']:
                try:
                    X = nl.Normalizer(X, strategy=preprocessing_plan['normalizer'], verbose=False).transform()
                    preprocessing_log.append(f"数据标准化: {preprocessing_plan['normalizer']}")
                except Exception as e:
                    logging.error(f"数据标准化失败: {str(e)}")
                    preprocessing_log.append("数据标准化: 跳过")
            
            # 6. 特征选择 - 最后执行，因为需要目标变量
            if 'feature_selector' in preprocessing_plan and preprocessing_plan['feature_selector']:
                try:
                    original_features = len(X.columns)
                    X = fs.Feature_selector(
                        X, 
                        target=target, 
                        strategy=preprocessing_plan['feature_selector'], 
                        threshold=0.1, 
                        exclude=target,
                        verbose=False
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
            #logging.info(f"预处理完成: {len(preprocessing_log)} 个步骤")
            if logging.getLogger().level <= logging.DEBUG:
                for step in preprocessing_log:
                    logging.debug(f"  - {step}")
            
            return X
            
        except Exception as e:
            logging.error(f"预处理执行失败: {str(e)}")
            # 返回原始数据，确保算法可以继续运行
            return data.copy()


class NSGA2_Preprocessing:
    """
    基于NSGA-II的多目标预处理优化
    使用多个机器学习模型来评估预处理组合，找到帕累托最优的预处理管道
    """
    def __init__(self, data, target, generations=10, population_size=20, mutation_rate=0.2, 
                 crossover_rate=0.9, models=None):
        """
        初始化NSGA-II预处理优化器
        
        Parameters:
        -----------
        data : pd.DataFrame
            输入数据
        target : str
            目标变量名
        generations : int
            进化代数
        population_size : int
            种群大小（建议是4的倍数，便于非支配排序）
        mutation_rate : float
            变异率
        crossover_rate : float
            交叉率
        models : list
            用于评估的模型列表，如果为None则自动选择
        """
        self.data = data
        self.target = target
        self.task_type, self.target_encoder = self._detect_task_type(data, target)
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # 初始化编码器
        enc.global_encoder.train_on_data(data, ratio_threshold=0.5, count_threshold=8)
        
        # 获取预处理步骤
        self.preprocessing_steps = self._get_default_preprocessing_steps()
        
        # 设置评估模型
        self.models = models if models else self._get_default_models()
        
        # 帕累托前沿
        self.pareto_front = []
        self.history = []
        
    def _detect_task_type(self, data, target):
        """检测任务类型"""
        try:
            if data is None or data.empty or target not in data.columns:
                raise ValueError("数据为空或目标变量不存在")
            
            target_series = data[target].dropna()
            if len(target_series) == 0:
                raise ValueError("目标变量没有有效值")
            
            target_dtype = target_series.dtype
            n_unique = target_series.nunique()
            n_samples = len(target_series)
            
            logging.info(f"NSGA-II: 目标变量 '{target}': 类型={target_dtype}, 唯一值={n_unique}, 样本数={n_samples}")
            
            if target_dtype in ['object', 'category', 'string']:
                from sklearn.preprocessing import LabelEncoder
                target_encoder = LabelEncoder()
                target_encoder.fit(target_series)
                if n_unique == 2:
                    return 'binary_classification', target_encoder
                else:
                    return 'multiclass_classification', target_encoder
            
            elif target_dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                if n_unique == 2:
                    return 'binary_classification', None
                elif n_unique <= 8 and n_unique <= n_samples * 0.1:
                    return 'multiclass_classification', None
                else:
                    return 'regression', None
            else:
                logging.warning(f"不支持的数据类型: {target_dtype}，默认为回归任务")
                return 'regression', None
                
        except Exception as e:
            logging.error(f"任务类型检测失败: {str(e)}")
            return 'regression', None
    
    def _get_default_preprocessing_steps(self):
        """获取默认的预处理步骤"""
        if self.task_type == 'regression':
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'L1', 'IMP'],
            }
        else:  # classification
            return {
                'normalizer': ['', 'ZS', 'DS', 'Log10', 'MM'],
                'imputer': ['EM', 'MICE', 'MF', 'MEDIAN', 'RAND'],
                'outliers': ['', 'ZSB', 'IQR'],
                'duplicate_detector': ['', 'ED'],
                'feature_selector': ['', 'MR', 'VAR', 'LC', 'Tree', 'WR'],
            }
    
    def _get_default_models(self):
        """获取默认的评估模型"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        
        if self.task_type == 'regression':
            return {
                'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'Ridge': Ridge(random_state=42)
            }
        else:  # classification
            return {
                'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=300, random_state=42)
            }
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for strategies in self.preprocessing_steps.values():
                if len(strategies) > 0:
                    chromosome.append(random.randint(0, len(strategies)-1))
                else:
                    chromosome.append(0)
            population.append(chromosome)
        return population
    
    def decode_chromosome(self, chromosome):
        """解码染色体为预处理方案"""
        decoded = {}
        index = 0
        for step, strategies in self.preprocessing_steps.items():
            if index < len(chromosome) and len(strategies) > 0:
                strategy_idx = min(chromosome[index], len(strategies)-1)
                decoded[step] = strategies[strategy_idx]
            else:
                decoded[step] = strategies[0] if strategies else ''
            index += 1
        return decoded
    
    def execute_preprocessing_plan(self, data, target, preprocessing_plan):
        """执行预处理计划（简化版本，直接调用Preprocessing类的方法）"""
        preprocessing = Preprocessing(data, target)
        return preprocessing.execute_preprocessing_plan(data, target, preprocessing_plan)
    
    def evaluate_individual(self, chromosome):
        """
        评估个体的多个目标
        
        Returns:
        --------
        objectives : list
            [model1_score, model2_score, model3_score, -preprocessing_time, -complexity]
        """
        plan = self.decode_chromosome(chromosome)
        objectives = []
        
        try:
            start_time = time.time()
            
            # 执行预处理
            processed_data = self.execute_preprocessing_plan(self.data.copy(), self.target, plan)
            
            # 分离特征和目标
            if self.target in processed_data.columns:
                X = processed_data.drop(columns=[self.target])
                y = processed_data[self.target]
            else:
                logging.error(f"目标变量 {self.target} 不在预处理后的数据中")
                return [-float('inf')] * (len(self.models) + 2)
            
            # 确保只使用数值特征
            X = X.select_dtypes(include=[np.number])
            
            if X.empty or len(X.columns) == 0:
                logging.warning("预处理后没有可用的数值特征")
                return [-float('inf')] * (len(self.models) + 2)
            
            # 使用多个模型评估
            for model_name, model in self.models.items():
                try:
                    if self.task_type == 'regression':
                        scores = cross_val_score(model, X, y, cv=3, 
                                                scoring='neg_mean_squared_error', n_jobs=1)
                        score = -np.mean(scores)  # 转换为正值，越小越好，但我们要最大化，所以取负
                        score = -score  # 再次取负，让越小越好变成越大越好
                    else:
                        scores = cross_val_score(model, X, y, cv=3, 
                                                scoring='accuracy', n_jobs=1)
                        score = np.mean(scores)
                    
                    objectives.append(score)
                    if logging.getLogger().level <= logging.DEBUG:
                        logging.debug(f"{model_name} 得分: {score:.4f}")
                    
                except Exception as e:
                    logging.error(f"模型 {model_name} 评估失败: {str(e)}")
                    objectives.append(-float('inf'))
            
            # 预处理时间（作为最小化目标，所以取负）
            preprocessing_time = time.time() - start_time
            objectives.append(-preprocessing_time)
            
            # 复杂度（非空步骤数量，作为最小化目标，所以取负）
            complexity = sum(1 for v in plan.values() if v and v != '')
            objectives.append(-complexity)
            
            if logging.getLogger().level <= logging.DEBUG:
                logging.debug(f"个体评估: 模型得分={objectives[:-2]}, 时间={preprocessing_time:.2f}s, 复杂度={complexity}")
            
        except Exception as e:
            logging.error(f"个体评估失败: {str(e)}")
            objectives = [-float('inf')] * (len(self.models) + 2)
        
        return objectives
    
    def fast_non_dominated_sort(self, population, objectives_list):
        """
        快速非支配排序
        
        Returns:
        --------
        fronts : list of lists
            每个前沿包含个体的索引
        """
        n = len(population)
        domination_count = [0] * n  # 支配该个体的个体数量
        dominated_solutions = [[] for _ in range(n)]  # 该个体支配的个体列表
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                
                if self._dominates(objectives_list[p], objectives_list[q]):
                    dominated_solutions[p].append(q)
                elif self._dominates(objectives_list[q], objectives_list[p]):
                    domination_count[p] += 1
            
            if domination_count[p] == 0:
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            if len(next_front) > 0:
                fronts.append(next_front)
            else:
                break
        
        return fronts[:-1] if len(fronts) > 1 and len(fronts[-1]) == 0 else fronts
    
    def _dominates(self, obj1, obj2):
        """
        判断obj1是否支配obj2
        支配条件：obj1在所有目标上不差于obj2，且至少在一个目标上严格优于obj2
        """
        better_in_any = False
        for o1, o2 in zip(obj1, obj2):
            if o1 < o2:  # obj1在这个目标上更差
                return False
            if o1 > o2:  # obj1在这个目标上更好
                better_in_any = True
        return better_in_any
    
    def calculate_crowding_distance(self, front_indices, objectives_list):
        """
        计算拥挤度距离
        
        Returns:
        --------
        distances : dict
            {个体索引: 拥挤度距离}
        """
        distances = {idx: 0 for idx in front_indices}
        
        if len(front_indices) <= 2:
            for idx in front_indices:
                distances[idx] = float('inf')
            return distances
        
        n_objectives = len(objectives_list[0])
        
        for m in range(n_objectives):
            # 按第m个目标排序
            sorted_indices = sorted(front_indices, 
                                  key=lambda idx: objectives_list[idx][m])
            
            # 边界点设置为无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # 获取目标值范围
            obj_min = objectives_list[sorted_indices[0]][m]
            obj_max = objectives_list[sorted_indices[-1]][m]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # 计算中间点的拥挤度距离
            for i in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[i]
                if distances[idx] != float('inf'):
                    distances[idx] += (objectives_list[sorted_indices[i+1]][m] - 
                                     objectives_list[sorted_indices[i-1]][m]) / obj_range
        
        return distances
    
    def crossover(self, parent1, parent2):
        """单点交叉"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, chromosome):
        """变异操作"""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                strategies = list(self.preprocessing_steps.values())[i]
                if len(strategies) > 0:
                    mutated[i] = random.randint(0, len(strategies)-1)
        return mutated
    
    def run(self):
        """
        运行NSGA-II算法
        
        Returns:
        --------
        pareto_solutions : list
            帕累托最优解列表，每个解包含 (chromosome, objectives, decoded_plan)
        """
        logging.info("="*60)
        logging.info("开始NSGA-II多目标预处理优化")
        logging.info(f"种群大小: {self.population_size}, 代数: {self.generations}")
        logging.info(f"评估模型: {list(self.models.keys())}")
        logging.info("="*60)
        
        # 初始化种群
        population = self.initialize_population()
        
        for gen in range(self.generations):
            logging.info(f"\n{'='*60}")
            logging.info(f"第 {gen+1}/{self.generations} 代")
            logging.info(f"{'='*60}")
            
            # 评估所有个体
            objectives_list = []
            for i, chromo in enumerate(population):
                obj = self.evaluate_individual(chromo)
                objectives_list.append(obj)
                if logging.getLogger().level <= logging.DEBUG:
                    logging.debug(f"个体 {i+1}: {obj}")
            
            # 非支配排序
            fronts = self.fast_non_dominated_sort(population, objectives_list)
            logging.info(f"非支配前沿数量: {len(fronts)}, 第一前沿个体数: {len(fronts[0])}")
            
            # 计算拥挤度距离
            all_distances = {}
            for front in fronts:
                distances = self.calculate_crowding_distance(front, objectives_list)
                all_distances.update(distances)
            
            # 生成子代
            offspring = []
            while len(offspring) < self.population_size:
                # 锦标赛选择（基于非支配排序和拥挤度距离）
                parent1_idx = self._tournament_select(population, fronts, all_distances)
                parent2_idx = self._tournament_select(population, fronts, all_distances)
                
                # 交叉
                child1, child2 = self.crossover(population[parent1_idx], 
                                               population[parent2_idx])
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            # 合并父代和子代
            combined_population = population + offspring[:self.population_size]
            combined_objectives = objectives_list + [self.evaluate_individual(c) for c in offspring[:self.population_size]]
            
            # 对合并种群进行非支配排序
            combined_fronts = self.fast_non_dominated_sort(combined_population, combined_objectives)
            
            # 选择下一代
            new_population = []
            new_objectives = []
            
            for front in combined_fronts:
                if len(new_population) + len(front) <= self.population_size:
                    for idx in front:
                        new_population.append(combined_population[idx])
                        new_objectives.append(combined_objectives[idx])
                else:
                    # 需要从当前前沿中选择部分个体
                    remaining = self.population_size - len(new_population)
                    distances = self.calculate_crowding_distance(front, combined_objectives)
                    sorted_front = sorted(front, key=lambda idx: distances[idx], reverse=True)
                    
                    for idx in sorted_front[:remaining]:
                        new_population.append(combined_population[idx])
                        new_objectives.append(combined_objectives[idx])
                    break
            
            population = new_population
            objectives_list = new_objectives
            
            # 记录历史
            self.history.append({
                'generation': gen + 1,
                'pareto_front_size': len(fronts[0]),
                'best_objectives': [objectives_list[i] for i in fronts[0][:3]]
            })
        
        # 最终的非支配排序
        fronts = self.fast_non_dominated_sort(population, objectives_list)
        
        # 提取帕累托前沿
        self.pareto_front = []
        for idx in fronts[0]:
            solution = {
                'chromosome': population[idx],
                'objectives': objectives_list[idx],
                'plan': self.decode_chromosome(population[idx])
            }
            self.pareto_front.append(solution)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"NSGA-II优化完成！")
        logging.info(f"帕累托前沿解的数量: {len(self.pareto_front)}")
        logging.info(f"{'='*60}\n")
        
        # 打印帕累托前沿解
        for i, sol in enumerate(self.pareto_front):
            logging.info(f"解 {i+1}:")
            logging.info(f"  目标值: {sol['objectives']}")
            logging.info(f"  预处理方案: {sol['plan']}")
        
        return self.pareto_front
    
    def _tournament_select(self, population, fronts, distances, tournament_size=2):
        """锦标赛选择"""
        candidates = random.sample(range(len(population)), tournament_size)
        
        # 找到每个候选者所在的前沿
        candidate_fronts = []
        for cand in candidates:
            for i, front in enumerate(fronts):
                if cand in front:
                    candidate_fronts.append(i)
                    break
        
        # 选择前沿等级最小的
        best_front = min(candidate_fronts)
        best_candidates = [candidates[i] for i, f in enumerate(candidate_fronts) if f == best_front]
        
        # 如果有多个最佳候选，选择拥挤度距离最大的
        if len(best_candidates) > 1:
            return max(best_candidates, key=lambda idx: distances[idx])
        else:
            return best_candidates[0]
    
    def get_best_solution(self, preference='balanced'):
        """
        从帕累托前沿中选择一个解
        
        Parameters:
        -----------
        preference : str
            'balanced': 选择综合性能最好的
            'fast': 选择预处理时间最短的
            'simple': 选择复杂度最低的
            'accurate': 选择模型性能最好的
        
        Returns:
        --------
        best_solution : dict
            最佳解
        """
        if not self.pareto_front:
            logging.error("帕累托前沿为空，请先运行run()方法")
            return None
        
        if preference == 'balanced':
            # 选择综合得分最高的（使用归一化后的加权和）
            scores = []
            for sol in self.pareto_front:
                obj = sol['objectives']
                # 归一化（简单方法：使用min-max）
                model_scores = obj[:-2]  # 前几个是模型得分
                normalized_score = np.mean(model_scores)
                scores.append(normalized_score)
            best_idx = np.argmax(scores)
            
        elif preference == 'fast':
            # 选择预处理时间最短的
            times = [sol['objectives'][-2] for sol in self.pareto_front]
            best_idx = np.argmax(times)  # 因为存储的是负值
            
        elif preference == 'simple':
            # 选择复杂度最低的
            complexities = [sol['objectives'][-1] for sol in self.pareto_front]
            best_idx = np.argmax(complexities)  # 因为存储的是负值
            
        elif preference == 'accurate':
            # 选择平均模型性能最好的
            avg_scores = [np.mean(sol['objectives'][:-2]) for sol in self.pareto_front]
            best_idx = np.argmax(avg_scores)
        
        else:
            logging.warning(f"未知的preference: {preference}，使用balanced")
            return self.get_best_solution('balanced')
        
        return self.pareto_front[best_idx]
    
    def get_processed_data(self, solution=None):
        """
        使用指定的解或最佳解处理数据
        
        Parameters:
        -----------
        solution : dict or None
            如果为None，自动选择balanced策略的最佳解
        
        Returns:
        --------
        processed_data : pd.DataFrame
            处理后的数据
        """
        if solution is None:
            solution = self.get_best_solution('balanced')
        
        if solution is None:
            logging.error("无法获取解决方案")
            return self.data.copy()
        
        plan = solution['plan']
        processed_data = self.execute_preprocessing_plan(self.data.copy(), self.target, plan)
        
        return processed_data


if __name__ == "__main__":
    import sys
    
    # 读取数据
    data = pd.read_csv('datasets/titanic_train.csv')
    target = 'Survived'
    
    print("选择优化方法:")
    print("1. 单目标遗传算法 (原方法)")
    print("2. NSGA-II多目标优化 (新方法)")
    
    choice = input("请选择 (1/2，默认2): ").strip() or "2"
    
    if choice == "1":
        print("\n=== 使用单目标遗传算法 ===")
        # 实例化预处理遗传算法类
        ga = Preprocessing(
            data=data,
            target=target,
            generations=3,
            population_size=4,
            mutation_rate=0.3,
            elite_size=1
        )
        
        # 运行遗传算法
        best_plan, best_score = ga.run()
        
        print("\n最佳预处理方案：", best_plan)
        print("最佳得分：", best_score)
        
        # 获取处理后的数据
        processed_data = ga.get_processed_data()
        print(f"处理后数据形状: {processed_data.shape}")
        
    else:
        print("\n=== 使用NSGA-II多目标优化 ===")
        # 实例化NSGA-II优化器
        nsga2 = NSGA2_Preprocessing(
            data=data,
            target=target,
            generations=5,
            population_size=12,
            mutation_rate=0.3,
            crossover_rate=0.9
        )
        
        # 运行NSGA-II
        pareto_front = nsga2.run()
        
        print(f"\n找到 {len(pareto_front)} 个帕累托最优解")
        
        # 显示不同策略的最佳解
        strategies = ['balanced', 'accurate', 'fast', 'simple']
        print("\n不同偏好的推荐方案:")
        print("-" * 80)
        
        for strategy in strategies:
            best_sol = nsga2.get_best_solution(strategy)
            if best_sol:
                print(f"\n{strategy.upper()} 策略:")
                print(f"  预处理方案: {best_sol['plan']}")
                print(f"  目标值: {best_sol['objectives']}")
        
        # 使用balanced策略获取处理后的数据
        print("\n使用 BALANCED 策略处理数据...")
        processed_data = nsga2.get_processed_data()
        print(f"处理后数据形状: {processed_data.shape}")

    print("处理后数据的形状：", processed_data.shape)
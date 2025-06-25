import logging

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

class Preprocessing:
    def __init__(self, data, target, generations=5, population_size=5, mutation_rate=0.2, elite_size=2):
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
        Xy = self.data.copy()
        Xy = self.execute_preprocessing_plan(Xy, self.target, self.best_plan)
        return Xy
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
if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('datasets/house_train.csv')
    data['SalePrice'] = np.log1p(data['SalePrice'])
    # 设定目标变量
    target = 'SalePrice'

    # 实例化预处理遗传算法类
    ga = Preprocessing(
        data=data,
        target=target,
        task_type='regression',  # 或 'binary_classification', 'multiclass_classification'
        generations=3,           # 迭代次数可以适当调小加快测试
        population_size=4,       # 种群数量
        mutation_rate=0.3,       # 变异率
        elite_size=1             # 精英保留数
        )

    # 运行遗传算法，自动搜索最佳预处理方案
    best_plan, best_score = ga.run()

    print("最佳预处理方案：", best_plan)
    print("最佳得分：", best_score)

    # 获取经过最佳预处理方案处理后的数据
    processed_data = ga.get_processed_data()
    print("处理后数据的形状：", processed_data.shape)
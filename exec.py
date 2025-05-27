import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import Timer, path_from_split
from amlb.results import save_predictions
from .Auto_ga import GeneticAlgorithm

def run(dataset: Dataset, config: TaskConfig):
    """运行 AutoGA 框架进行模型训练和预测"""
    
    # 获取配置参数
    params = config.framework_params
    generations = params.get('generations', 20)
    population_size = params.get('population_size', 50)
    mutation_rate = params.get('mutation_rate', 0.2)
    elite_size = params.get('elite_size', 2)
    use_prediction = params.get('use_prediction', True)
    cv_scoring = params.get('cv_scoring', 'auto')
    n_jobs = params.get('n_jobs', -1)
    time_limit = params.get('time_limit', None)
    tournament_size = params.get('tournament_size', 3)
    
    # 准备数据
    X_train = dataset.train.X
    y_train = dataset.train.y
    X_test = dataset.test.X
    
    # 合并训练数据用于遗传算法
    train_data = X_train.copy()
    train_data[dataset.target] = y_train
    
    # 初始化遗传算法
    ga = GeneticAlgorithm(use_prediction=use_prediction, cv_scoring=cv_scoring)
    
    # 训练模型
    with Timer() as training:
        best_config, best_score, history, avg_history, best_model = ga.run(
            data=train_data,
            target=dataset.target,
            generations=generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            n_jobs=n_jobs,
            time_limit=time_limit,
            tournament_size=tournament_size
        )
    
    # 预测
    with Timer() as predict:
        # 使用最佳配置进行预处理
        processed_test = ga.execute_preprocessing_plan(
            X_test.copy(),
            dataset.target,
            ga.decode_chromosome(best_config)['preprocessing']
        )
        predictions = best_model.predict(processed_test)
    
    # 如果是分类任务,将预测结果转换回原始标签
    if config.type == 'classification':
        le = LabelEncoder()
        le.fit(y_train)
        predictions = le.inverse_transform(predictions)
    
    # 保存预测结果
    save_predictions(dataset=dataset,
                    output_file=config.output_predictions_file,
                    predictions=predictions)
    
    # 返回结果
    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
        best_score=best_score
    ) 
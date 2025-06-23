import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import time
import logging
from Auto_ga import GeneticAlgorithm
import load_data as ld
from tpot import TPOTClassifier, TPOTRegressor
import h2o
from h2o.automl import H2OAutoML
from flaml import AutoML

def evaluate_auto_ml_models(X_train, X_test, y_train, y_test, task_type='classification', time_limit=300):
    """
    评估各种自动机器学习框架的性能
    """
    results = {}
    
    # 1. TPOT
    try:
        print("正在运行 TPOT...")
        start_time = time.time()
        if task_type == 'classification':
            tpot = TPOTClassifier(
                generations=5,
                population_size=20,
                cv=5,
                verbosity=2,
                n_jobs=-1,
                max_time_mins=time_limit//60
            )
            tpot.fit(X_train, y_train)
            # 在测试集上评估，而不是训练集
            score = tpot.score(X_test, y_test)
        else:
            tpot = TPOTRegressor(
                generations=5,
                population_size=20,
                cv=5,
                verbosity=2,
                n_jobs=-1,
                max_time_mins=time_limit//60
            )
            tpot.fit(X_train, y_train)
            # 在测试集上评估
            y_pred = tpot.predict(X_test)
            score = -mean_squared_error(y_test, y_pred)
        
        training_time = time.time() - start_time
        results['TPOT'] = {
            'score': score,
            'training_time': training_time,
            'best_model': tpot.fitted_pipeline_
        }
    except Exception as e:
        print(f"TPOT 运行失败: {str(e)}")
    
    # 2. H2O AutoML
    try:
        print("正在运行 H2O AutoML...")
        h2o.init()
        start_time = time.time()
        
        # 转换为H2O格式
        train = h2o.H2OFrame(pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='target')], axis=1))
        test = h2o.H2OFrame(pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name='target')], axis=1))
        
        if task_type == 'classification':
            aml = H2OAutoML(
                max_runtime_secs=time_limit,
                sort_metric='AUC',
                seed=1
            )
            aml.train(x=list(range(X_train.shape[1])), y='target', training_frame=train)
            # 在测试集上评估
            score = aml.leader.model_performance(test).auc()
        else:
            aml = H2OAutoML(
                max_runtime_secs=time_limit,
                sort_metric='RMSE',
                seed=1
            )
            aml.train(x=list(range(X_train.shape[1])), y='target', training_frame=train)
            # 在测试集上评估
            score = -aml.leader.model_performance(test).rmse()
        
        training_time = time.time() - start_time
        results['H2O AutoML'] = {
            'score': score,
            'training_time': training_time,
            'best_model': aml.leader
        }
        h2o.cluster().shutdown()
    except Exception as e:
        print(f"H2O AutoML 运行失败: {str(e)}")
    
    # 3. FLAML
    try:
        print("正在运行 FLAML...")
        start_time = time.time()
        
        automl = AutoML()
        automl_settings = {
            "time_budget": time_limit,
            "metric": 'accuracy' if task_type == 'classification' else 'rmse',
            "task": task_type,
            "log_file_name": "flaml.log"
        }
        
        automl.fit(X_train, y_train, **automl_settings)
        # 在测试集上评估，而不是训练集
        score = automl.score(X_test, y_test)
        
        training_time = time.time() - start_time
        results['FLAML'] = {
            'score': score,
            'training_time': training_time,
            'best_model': automl.model
        }
    except Exception as e:
        print(f"FLAML 运行失败: {str(e)}")
    
    return results

def compare_auto_ml_frameworks(data, target, task_type='classification', time_limit=300, test_size=0.2, random_state=42):
    """
    比较不同自动机器学习框架的性能
    """
    # 数据预处理和分离
    X = data.drop(columns=[target])
    y = data[target]
    
    # 分离训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == 'classification' else None
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 评估其他自动机器学习框架
    print("\n正在评估其他自动机器学习框架...")
    auto_ml_results = evaluate_auto_ml_models(X_train, X_test, y_train, y_test, task_type, time_limit)
    
    # 评估我们的自动机器学习模型
    print("\n正在运行我们的自动机器学习模型...")
    start_time = time.time()
    
    # 为遗传算法准备完整的数据集（包含训练和测试数据）
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    ga = GeneticAlgorithm(data=train_data, target=target, use_prediction=True)
    best_config, best_score, history, avg_history, best_model = ga.run(
        data=train_data,
        target=target,
        generations=20,
        population_size=20,
        n_jobs=-1
    )
    auto_ml_time = time.time() - start_time
    
    # 在测试集上评估我们的模型
    if best_model is not None:
        # 使用最佳配置在测试集上评估
        config = ga.decode_chromosome(best_config)
        processed_test_data = ga.execute_preprocessing_plan(test_data.copy(), target, config['preprocessing'])
        
        if task_type == 'classification':
            test_score = best_model.score(processed_test_data.drop(columns=[target]), processed_test_data[target])
        else:
            y_pred = best_model.predict(processed_test_data.drop(columns=[target]))
            test_score = -mean_squared_error(processed_test_data[target], y_pred)
    else:
        test_score = best_score  # 如果模型为None，使用训练时的分数
    
    # 准备结果报告
    print("\n=== 自动机器学习框架性能比较报告 ===")
    
    print("\n其他自动机器学习框架性能 (测试集):")
    for framework, metrics in auto_ml_results.items():
        print(f"\n{framework}:")
        if task_type == 'classification':
            print(f"准确率: {metrics['score']:.4f}")
        else:
            print(f"均方误差: {-metrics['score']:.4f}")
        print(f"训练时间: {metrics['training_time']:.2f}秒")
        print("最佳模型配置:")
        print(metrics['best_model'])
    
    print("\n我们的自动机器学习模型性能:")
    print(f"训练集得分: {best_score:.4f}")
    print(f"测试集得分: {test_score:.4f}")
    print(f"总运行时间: {auto_ml_time:.2f}秒")
    print("\n最佳配置:")
    decoded_config = ga.decode_chromosome(best_config)
    print("预处理步骤:")
    for step, method in decoded_config['preprocessing'].items():
        print(f"  {step}: {method}")
    print(f"模型: {decoded_config['model']}")
    print("超参数:")
    for param, value in decoded_config['hyperparameters'].items():
        print(f"  {param}: {value}")
    
    # 保存结果到日志
    logging.info("\n=== 自动机器学习框架性能比较报告 ===")
    logging.info(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    logging.info("\n其他自动机器学习框架性能 (测试集):")
    for framework, metrics in auto_ml_results.items():
        logging.info(f"\n{framework}:")
        if task_type == 'classification':
            logging.info(f"准确率: {metrics['score']:.4f}")
        else:
            logging.info(f"均方误差: {-metrics['score']:.4f}")
        logging.info(f"训练时间: {metrics['training_time']:.2f}秒")
        logging.info("最佳模型配置:")
        logging.info(str(metrics['best_model']))
    
    logging.info("\n我们的自动机器学习模型性能:")
    logging.info(f"训练集得分: {best_score:.4f}")
    logging.info(f"测试集得分: {test_score:.4f}")
    logging.info(f"总运行时间: {auto_ml_time:.2f}秒")
    logging.info("\n最佳配置:")
    logging.info("预处理步骤:")
    for step, method in decoded_config['preprocessing'].items():
        logging.info(f"  {step}: {method}")
    logging.info(f"模型: {decoded_config['model']}")
    logging.info("超参数:")
    for param, value in decoded_config['hyperparameters'].items():
        logging.info(f"  {param}: {value}")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        filename='auto_ml_comparison.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    
    # 加载数据
    data = ld.load_data("datasets/titanic_train.csv")
    target = "Survived"
    
    # 运行比较
    compare_auto_ml_frameworks(data, target, task_type='classification', time_limit=300) 
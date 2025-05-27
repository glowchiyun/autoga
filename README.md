# AutoGA

AutoGA 是一个基于遗传算法的自动化机器学习框架。它能够自动处理数据预处理、特征选择、模型选择和超参数调优等任务。

## 特性

- 自动数据预处理
  - 缺失值处理
  - 异常值检测
  - 数据标准化
  - 特征编码
- 智能特征选择
  - 基于遗传算法的特征重要性评估
  - 自动特征组合
- 模型选择与优化
  - 支持分类和回归任务
  - 自动模型选择
  - 超参数优化
- 可扩展性
  - 模块化设计
  - 易于扩展新功能

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```python
from autoga import AutoGA

# 创建 AutoGA 实例
automl = AutoGA(
    task_type='classification',  # 或 'regression'
    generations=20,
    population_size=50,
    mutation_rate=0.2,
    elite_size=2
)

# 训练模型
automl.fit(X_train, y_train)

# 预测
predictions = automl.predict(X_test)
```

## 主要模块

- `Auto_ga.py`: 核心遗传算法实现
- `classifier.py`: 分类器实现
- `regressor.py`: 回归器实现
- `feature_selector.py`: 特征选择器
- `imputer.py`: 缺失值处理
- `normalizer.py`: 数据标准化
- `outlier_detector.py`: 异常值检测
- `encoder.py`: 特征编码
- `clusterer.py`: 聚类分析
- `duplicate_detector.py`: 重复值检测
- `load_data.py`: 数据加载
- `make_data.py`: 数据生成

## 依赖

- Python >= 3.9
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- deap >= 1.3.1

## 贡献

欢迎提交 Pull Request 或创建 Issue 来帮助改进项目。

## 许可证

MIT License 
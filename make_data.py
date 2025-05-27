import numpy as np
from sklearn.datasets import make_classification
import pandas as pd

def generate_similar_datasets(n_samples=1000, n_features=10, n_informative=5, n_redundant=3, n_classes=2, noise_level=0.1):
    """
    生成两个具有相似性的分类数据集，包含缺失值、离群点和需要归一化处理的特征。
    
    参数:
    - n_samples: 每个数据集的样本数量
    - n_features: 每个数据集的特征数量
    - n_informative: 信息特征的数量
    - n_redundant: 冗余特征的数量
    - n_classes: 分类的类别数量
    - noise_level: 噪声水平
    
    返回:
    - source_data, source_target: 源数据集
    - target_data, target_target: 目标数据集
    """
    
    # 生成源数据集
    X_source, y_source = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=42
    )
    
    # 生成目标数据集，添加一些噪声和变化
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, X_source.shape)
    X_target = X_source + noise
    
    # 添加一些新的特征
    new_features = np.random.rand(n_samples, 2)
    X_target = np.hstack((X_target, new_features))
    
    # 添加一些新的标签噪声
    flip_mask = np.random.rand(n_samples) < noise_level
    y_target = y_source.copy()
    y_target[flip_mask] = 1 - y_target[flip_mask]
    
    return (X_source, y_source), (X_target, y_target)

def add_missing_values(X, missing_rate=0.1):
    """
    添加缺失值。
    
    参数:
    - X: 特征矩阵
    - missing_rate: 缺失值比例
    
    返回:
    - X_with_missing: 包含缺失值的特征矩阵
    """
    X_with_missing = X.copy()
    n_samples, n_features = X.shape
    missing_mask = np.random.rand(n_samples, n_features) < missing_rate
    X_with_missing[missing_mask] = np.nan
    return X_with_missing

def add_outliers(X, outlier_rate=0.05):
    """
    添加离群点。
    
    参数:
    - X: 特征矩阵
    - outlier_rate: 离群点比例
    
    返回:
    - X_with_outliers: 包含离群点的特征矩阵
    """
    X_with_outliers = X.copy()
    n_samples, n_features = X.shape
    outlier_mask = np.random.rand(n_samples, n_features) < outlier_rate
    # 将离群点设置为远大于正常值的值
    X_with_outliers[outlier_mask] = np.random.normal(10, 5, size=(outlier_mask.sum(),))
    return X_with_outliers

def add_unnormalized_features(X, scale_factor=10):
    """
    添加需要归一化处理的特征。
    
    参数:
    - X: 特征矩阵
    - scale_factor: 特征值的缩放因子
    
    返回:
    - X_with_unnormalized: 包含需要归一化处理的特征矩阵
    """
    X_with_unnormalized = X.copy()
    # 将某些特征的值放大，使其需要归一化
    for i in range(X.shape[1]):
        if i % 2 == 0:  # 假设每隔一个特征需要归一化
            X_with_unnormalized[:, i] *= scale_factor
    return X_with_unnormalized

# 使用示例
if __name__ == "__main__":
    # 生成两个相似的数据集
    (X_source, y_source), (X_target, y_target) = generate_similar_datasets(
        n_samples=1000,
        n_features=12,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        noise_level=0.1
    )
    
    # 添加缺失值
    X_source_with_missing = add_missing_values(X_source, missing_rate=0.1)
    X_target_with_missing = add_missing_values(X_target, missing_rate=0.1)
    
    # 添加离群点
    X_source_with_outliers = add_outliers(X_source_with_missing, outlier_rate=0.05)
    X_target_with_outliers = add_outliers(X_target_with_missing, outlier_rate=0.05)
    
    # 添加需要归一化处理的特征
    X_source_final = add_unnormalized_features(X_source_with_outliers, scale_factor=10)
    X_target_final = add_unnormalized_features(X_target_with_outliers, scale_factor=10)
    
    # 将特征和目标值合并为一个DataFrame
    source_df = pd.DataFrame(X_source_final)
    source_df['target'] = y_source
    target_df = pd.DataFrame(X_target_final)
    target_df['target'] = y_target
    
    # 打印数据集信息
    print("源数据集形状:", source_df.shape)
    print("目标数据集形状:", target_df.shape)
    
    # 将数据集保存为CSV文件
    source_df.to_csv("datasets/source_dataset_with_issues.csv", index=False)
    target_df.to_csv("datasets/target_dataset_with_issues.csv", index=False)
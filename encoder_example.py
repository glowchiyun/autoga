import pandas as pd
from encoder import Encoder
import numpy as np

def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    n_samples = 1000
    
    # 创建分类特征
    categories = ['A', 'B', 'C', 'D', 'E']
    binary_cats = ['Yes', 'No']
    
    data = {
        'numeric_feature': np.random.normal(0, 1, n_samples),
        'binary_feature': np.random.choice(binary_cats, n_samples),
        'categorical_feature': np.random.choice(categories, n_samples),
        'high_cardinality': [f'category_{i}' for i in np.random.randint(0, 100, n_samples)],
        'constant_feature': ['same_value'] * n_samples
    }
    
    return pd.DataFrame(data)

def main():
    print("=== 编码器保存和加载示例 ===\n")
    
    # 1. 创建训练数据
    print("1. 创建训练数据...")
    train_data = create_sample_data()
    print(f"训练数据形状: {train_data.shape}")
    print(f"分类特征: {list(train_data.select_dtypes(include=['object']).columns)}")
    print()
    
    # 2. 训练编码器
    print("2. 训练编码器...")
    encoder = Encoder(ratio_threshold=0.3, count_threshold=20)
    encoded_train = encoder.fit_transform(train_data)
    
    # 显示编码摘要
    summary = encoder.get_encoding_summary()
    print(f"编码特征总数: {summary['total_features']}")
    print(f"Label编码特征: {summary['label_encoded']}")
    print(f"Binary编码特征: {summary['binary_encoded']}")
    print(f"跳过特征: {summary['skipped_features']}")
    print()
    
    # 显示编码详情
    print("编码详情:")
    for col, info in summary['encoding_details'].items():
        print(f"  {col}: {info['编码方式']} -> {info['输出特征数']}个特征")
    
    if summary['skipped_details']:
        print("\n跳过特征详情:")
        for col, reason in summary['skipped_details'].items():
            print(f"  {col}: {reason}")
    print()
    
    # 3. 保存编码器
    print("3. 保存编码器...")
    encoder.save_encoder("my_encoder")
    print()
    
    # 4. 创建新的测试数据（可能包含新的类别值）
    print("4. 创建测试数据...")
    test_data = create_sample_data()
    # 添加一些新的类别值来测试
    test_data.loc[0, 'categorical_feature'] = 'F'  # 新类别
    test_data.loc[1, 'binary_feature'] = 'Maybe'   # 新类别
    print(f"测试数据形状: {test_data.shape}")
    print()
    
    # 5. 加载编码器并应用到新数据
    print("5. 加载编码器并应用到新数据...")
    new_encoder = Encoder()
    new_encoder.load_encoder("my_encoder")
    
    # 应用相同的编码流程
    encoded_test = new_encoder.transform(test_data)
    print(f"编码后测试数据形状: {encoded_test.shape}")
    print()
    
    # 6. 比较编码结果
    print("6. 编码结果比较:")
    print(f"训练数据编码后特征数: {encoded_train.shape[1]}")
    print(f"测试数据编码后特征数: {encoded_test.shape[1]}")
    print()
    
    # 7. 显示编码后的列名
    print("编码后的特征列名:")
    print(f"训练数据: {list(encoded_train.columns)}")
    print(f"测试数据: {list(encoded_test.columns)}")
    print()
    
    # 8. 验证编码一致性
    print("8. 验证编码一致性...")
    # 检查相同的特征是否使用相同的编码方式
    train_categorical = train_data.select_dtypes(include=['object']).columns
    test_categorical = test_data.select_dtypes(include=['object']).columns
    
    common_features = set(train_categorical) & set(test_categorical)
    print(f"共同分类特征: {list(common_features)}")
    
    for feature in common_features:
        if feature in new_encoder.encoding_info:
            strategy = new_encoder.encoding_info[feature]['strategy']
            print(f"  {feature}: 使用 {strategy} 编码")
        else:
            print(f"  {feature}: 被跳过")
    
    print("\n=== 示例完成 ===")

if __name__ == "__main__":
    main() 
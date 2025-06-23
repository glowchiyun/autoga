import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder
import math
import pickle
import json
import os

# 全局编码器管理器
class GlobalEncoderManager:
    """全局编码器管理器，用于在整个ML流程中复用编码器"""
    
    def __init__(self):
        self.encoders = {}
        self.encoding_info = {}
        self.skipped_features = {}
        self.is_trained = False
        self.ratio_threshold = 0.5
        self.count_threshold = 50
        
    def train_on_data(self, dataset, ratio_threshold=0.5, count_threshold=50):
        """在数据集上训练编码器"""
        self.ratio_threshold = ratio_threshold
        self.count_threshold = count_threshold
        
        # 重置状态
        self.encoders = {}
        self.encoding_info = {}
        self.skipped_features = {}
        
        X = dataset.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            self.is_trained = True
            return
            
        # 对每个分类特征进行编码训练
        for col in categorical_cols:
            self._train_feature_encoder(X, col)
            
        self.is_trained = True
        print(f"全局编码器训练完成: 编码{len(self.encoding_info)}个特征，跳过{len(self.skipped_features)}个特征")
        
    def _train_feature_encoder(self, X, col):
        """训练单个特征的编码器"""
        try:
            strategy, n_features = self._get_encoding_strategy(X, col)
            
            if strategy is None:
                return
                
            # 记录编码信息
            self.encoding_info[col] = {
                'strategy': strategy,
                'n_unique_values': X[col].nunique(),
                'n_output_features': n_features,
                'ratio': X[col].nunique() / len(X)
            }
            
            # 训练编码器
            if strategy == 'LE':
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(X[col].astype(str))
            else:  # Binary Encoding
                self.encoders[col] = BinaryEncoder(cols=[col])
                self.encoders[col].fit(X[[col]])
                
        except Exception as e:
            print(f"训练特征 {col} 编码器时发生错误: {str(e)}")
            self.skipped_features[col] = f"编码错误: {str(e)}"
    
    def _check_thresholds(self, X, col):
        """检查特征是否满足编码条件"""
        n_unique = X[col].nunique()
        n_samples = len(X)
        ratio = n_unique / n_samples
            
        # 检查唯一值数量
        if n_unique == 1:
            self.skipped_features[col] = "只有一个唯一值"
            return False
            
        # 检查是否为数值型
        if pd.api.types.is_numeric_dtype(X[col]):
            self.skipped_features[col] = "数值型特征"
            return False
            
        # 检查唯一值比例
        if ratio >= self.ratio_threshold:
            self.skipped_features[col] = f"唯一值比例过高 ({ratio:.2%} >= {self.ratio_threshold:.2%})"
            return False
            
        # 检查唯一值数量
        if n_unique >= self.count_threshold:
            self.skipped_features[col] = f"唯一值数量过多 ({n_unique} >= {self.count_threshold})"
            return False
        
        return True
        
    def _get_encoding_strategy(self, X, col):
        """根据特征的唯一值数量确定编码策略"""
        if not self._check_thresholds(X, col):
            return None, 0
            
        n_unique = X[col].nunique()
        
        # 如果是二分类特征，使用Label Encoding
        if n_unique == 2:
            return 'LE', 1
            
        # 如果是多分类特征，使用Binary Encoding
        return 'BE', math.ceil(math.log2(n_unique))
    
    def transform_data(self, dataset):
        """使用训练好的编码器转换数据"""
        if not self.is_trained:
            raise ValueError("编码器尚未训练，请先调用train_on_data")
            
        X = dataset.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return X
            
        # 对每个分类特征进行编码
        for col in categorical_cols:
            # 如果特征在训练时被跳过，则跳过
            if col in self.skipped_features:
                continue
                
            # 如果特征在训练时被编码，则应用相同的编码
            if col in self.encoders:
                if self.encoding_info[col]['strategy'] == 'LE':
                    # Label Encoding - 处理新类别
                    try:
                        encoded = self.encoders[col].transform(X[col].astype(str))
                        X[col] = encoded
                    except ValueError as e:
                        # 处理新类别值
                        print(f"警告: 特征 {col} 包含新类别值，使用默认值填充")
                        # 将新类别值替换为最常见的类别
                        most_common = X[col].mode().iloc[0] if not X[col].mode().empty else X[col].iloc[0]
                        X[col] = X[col].replace(X[col].unique(), most_common)
                        encoded = self.encoders[col].transform(X[col].astype(str))
                        X[col] = encoded
                else:
                    # Binary Encoding - 处理新类别
                    try:
                        encoded = self.encoders[col].transform(X[[col]])
                        X = pd.concat([X.drop(col, axis=1), encoded], axis=1)
                    except Exception as e:
                        # 处理新类别值
                        print(f"警告: 特征 {col} 包含新类别值，使用默认值填充")
                        # 将新类别值替换为最常见的类别
                        most_common = X[col].mode().iloc[0] if not X[col].mode().empty else X[col].iloc[0]
                        X[col] = X[col].replace(X[col].unique(), most_common)
                        encoded = self.encoders[col].transform(X[[col]])
                        X = pd.concat([X.drop(col, axis=1), encoded], axis=1)
            else:
                # 新特征，跳过
                print(f"警告: 特征 {col} 在训练时未出现，跳过编码")
                    
        return X
    
    def get_encoding_summary(self):
        """获取编码摘要信息"""
        summary = {
            'total_features': len(self.encoding_info),
            'label_encoded': sum(1 for info in self.encoding_info.values() if info['strategy'] == 'LE'),
            'binary_encoded': sum(1 for info in self.encoding_info.values() if info['strategy'] == 'BE'),
            'skipped_features': len(self.skipped_features),
            'encoding_details': {},
            'skipped_details': self.skipped_features
        }
        
        # 添加每个特征的详细信息
        for col, info in self.encoding_info.items():
            summary['encoding_details'][col] = {
                '编码方式': 'Label Encoding' if info['strategy'] == 'LE' else 'Binary Encoding',
                '唯一值数量': info['n_unique_values'],
                '唯一值比例': f"{info['ratio']:.2%}",
                '输出特征数': info['n_output_features']
            }
            
        return summary

# 全局编码器实例
global_encoder = GlobalEncoderManager()

class Encoder:
    """
    自动编码类，根据特征的唯一值数量和比例自动选择编码方式：
    - 数值型特征：保持不变
    - 二分类特征（2个唯一值）：使用Label Encoding
    - 多分类特征（>2个唯一值）：使用Binary Encoding
    
    Parameters:
    ----------
    dataset: DataFrame
        输入数据集
    ratio_threshold: float, default=0.5
        唯一值比例阈值，当唯一值数量/样本数量 < ratio_threshold 时才进行编码
    count_threshold: int, default=50
        唯一值数量阈值，当唯一值数量 < count_threshold 时才进行编码
    use_global_encoder: bool, default=True
        是否使用全局编码器（推荐用于自动ML流程）
    """
    def __init__(self, dataset=None, ratio_threshold=0.5, count_threshold=50, use_global_encoder=True):
        self.dataset = dataset
        self.ratio_threshold = ratio_threshold
        self.count_threshold = count_threshold
        self.use_global_encoder = use_global_encoder
        self.encoders = {}  # 存储每个特征的编码器
        self.encoding_info = {}  # 记录每个特征的编码信息
        self.skipped_features = {}  # 记录被跳过的特征及原因
        self.is_fitted = False  # 标记是否已经训练过
        
    def _check_thresholds(self, X, col):
        """
        检查特征是否满足编码条件
        """
        n_unique = X[col].nunique()
        n_samples = len(X)
        ratio = n_unique / n_samples
            
        # 检查唯一值数量
        if n_unique == 1:
            self.skipped_features[col] = "只有一个唯一值"
            return False
            
        # 检查是否为数值型
        if pd.api.types.is_numeric_dtype(X[col]):
            self.skipped_features[col] = "数值型特征"
            return False
            
        # 检查唯一值比例
        if ratio >= self.ratio_threshold:
            self.skipped_features[col] = f"唯一值比例过高 ({ratio:.2%} >= {self.ratio_threshold:.2%})"
            return False
            
        # 检查唯一值数量
        if n_unique >= self.count_threshold:
            self.skipped_features[col] = f"唯一值数量过多 ({n_unique} >= {self.count_threshold})"
            return False
        
        return True
        
    def _get_encoding_strategy(self, X, col):
        """
        根据特征的唯一值数量确定编码策略
        """
        if not self._check_thresholds(X, col):
            return None, 0
            
        n_unique = X[col].nunique()
        
        # 如果是二分类特征，使用Label Encoding
        if n_unique == 2:
            return 'LE', 1
            
        # 如果是多分类特征，使用Binary Encoding
        return 'BE', math.ceil(math.log2(n_unique))
        
    def _encode_feature(self, X, col):
        """
        对单个特征进行编码
        """
        try:
            strategy, n_features = self._get_encoding_strategy(X, col)
            
            if strategy is None:
                return None
                
            # 记录编码信息
            self.encoding_info[col] = {
                'strategy': strategy,
                'n_unique_values': X[col].nunique(),
                'n_output_features': n_features,
                'ratio': X[col].nunique() / len(X)
            }
            
            # 执行编码
            if strategy == 'LE':
                self.encoders[col] = LabelEncoder()
                encoded = self.encoders[col].fit_transform(X[col].astype(str))
                return encoded
            else:  # Binary Encoding
                self.encoders[col] = BinaryEncoder(cols=[col])
                encoded = self.encoders[col].fit_transform(X[[col]])
                return encoded
                
        except Exception as e:
            print(f"编码特征 {col} 时发生错误: {str(e)}")
            self.skipped_features[col] = f"编码错误: {str(e)}"
            return None
    
    def fit(self, dataset=None):
        """
        训练编码器（可选方法，transform方法会自动调用）
        """
        if dataset is not None:
            self.dataset = dataset
        
        if self.dataset is None:
            raise ValueError("需要提供数据集进行训练")
            
        # 如果使用全局编码器且全局编码器已训练，直接使用
        if self.use_global_encoder and global_encoder.is_trained:
            self.encoders = global_encoder.encoders.copy()
            self.encoding_info = global_encoder.encoding_info.copy()
            self.skipped_features = global_encoder.skipped_features.copy()
            self.is_fitted = True
            return self
            
        # 如果使用全局编码器但全局编码器未训练，先训练全局编码器
        if self.use_global_encoder and not global_encoder.is_trained:
            global_encoder.train_on_data(self.dataset, self.ratio_threshold, self.count_threshold)
            self.encoders = global_encoder.encoders.copy()
            self.encoding_info = global_encoder.encoding_info.copy()
            self.skipped_features = global_encoder.skipped_features.copy()
            self.is_fitted = True
            return self
            
        # 否则使用本地编码器
        # 重置状态
        self.encoders = {}
        self.encoding_info = {}
        self.skipped_features = {}
        
        X = self.dataset.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            self.is_fitted = True
            return self
            
        # 对每个分类特征进行编码训练
        for col in categorical_cols:
            self._encode_feature(X, col)
            
        self.is_fitted = True
        return self
            
    def transform(self, dataset=None):
        """
        对数据集进行编码转换
        """
        if dataset is not None:
            X = dataset.copy()
        else:
            if self.dataset is None:
                raise ValueError("需要提供数据集进行转换")
            X = self.dataset.copy()
        
        # 如果使用全局编码器且全局编码器已训练，直接使用全局编码器
        if self.use_global_encoder and global_encoder.is_trained:
            return global_encoder.transform_data(X)
        
        # 如果没有训练过，先进行训练
        if not self.is_fitted:
            self.fit(X)
            
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return X
            
        # 记录原始列名
        original_columns = set(X.columns)
        
        # 对每个分类特征进行编码
        for col in categorical_cols:
            # 如果特征在训练时被跳过，则跳过
            if col in self.skipped_features:
                continue
                
            # 如果特征在训练时被编码，则应用相同的编码
            if col in self.encoders:
                if self.encoding_info[col]['strategy'] == 'LE':
                    # Label Encoding - 处理新类别
                    try:
                        encoded = self.encoders[col].transform(X[col].astype(str))
                        X[col] = encoded
                    except ValueError as e:
                        # 处理新类别值
                        print(f"警告: 特征 {col} 包含新类别值，使用默认值填充")
                        # 将新类别值替换为最常见的类别
                        most_common = X[col].mode().iloc[0] if not X[col].mode().empty else X[col].iloc[0]
                        X[col] = X[col].replace(X[col].unique(), most_common)
                        encoded = self.encoders[col].transform(X[col].astype(str))
                        X[col] = encoded
                else:
                    # Binary Encoding - 处理新类别
                    try:
                        encoded = self.encoders[col].transform(X[[col]])
                        X = pd.concat([X.drop(col, axis=1), encoded], axis=1)
                    except Exception as e:
                        # 处理新类别值
                        print(f"警告: 特征 {col} 包含新类别值，使用默认值填充")
                        # 将新类别值替换为最常见的类别
                        most_common = X[col].mode().iloc[0] if not X[col].mode().empty else X[col].iloc[0]
                        X[col] = X[col].replace(X[col].unique(), most_common)
                        encoded = self.encoders[col].transform(X[[col]])
                        X = pd.concat([X.drop(col, axis=1), encoded], axis=1)
            else:
                # 新特征，需要重新训练
                encoded = self._encode_feature(X, col)
                if encoded is not None:
                    if isinstance(encoded, pd.DataFrame):  # Binary Encoding的结果
                        X = pd.concat([X.drop(col, axis=1), encoded], axis=1)
                    else:  # Label Encoding的结果
                        X[col] = encoded
                    
        return X
    
    def fit_transform(self, dataset=None):
        """
        训练编码器并转换数据集
        """
        if dataset is not None:
            self.dataset = dataset
            
        self.fit()
        return self.transform()
        
    def save_encoder(self, filepath):
        """
        保存编码器到文件
        
        Parameters:
        ----------
        filepath: str
            保存路径，不需要扩展名（会自动添加.pkl和.json）
        """
        if not self.is_fitted:
            raise ValueError("编码器尚未训练，无法保存")
            
        # 保存编码器对象
        encoder_file = f"{filepath}_encoders.pkl"
        with open(encoder_file, 'wb') as f:
            pickle.dump(self.encoders, f)
            
        # 保存编码信息和参数
        info_file = f"{filepath}_info.json"
        save_info = {
            'ratio_threshold': self.ratio_threshold,
            'count_threshold': self.count_threshold,
            'encoding_info': self.encoding_info,
            'skipped_features': self.skipped_features,
            'is_fitted': self.is_fitted
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(save_info, f, ensure_ascii=False, indent=2)
            
        print(f"编码器已保存到: {filepath}")
        
    def load_encoder(self, filepath):
        """
        从文件加载编码器
        
        Parameters:
        ----------
        filepath: str
            加载路径，不需要扩展名（会自动添加.pkl和.json）
        """
        # 加载编码器对象
        encoder_file = f"{filepath}_encoders.pkl"
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"编码器文件不存在: {encoder_file}")
            
        with open(encoder_file, 'rb') as f:
            self.encoders = pickle.load(f)
            
        # 加载编码信息和参数
        info_file = f"{filepath}_info.json"
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"信息文件不存在: {info_file}")
            
        with open(info_file, 'r', encoding='utf-8') as f:
            save_info = json.load(f)
            
        self.ratio_threshold = save_info['ratio_threshold']
        self.count_threshold = save_info['count_threshold']
        self.encoding_info = save_info['encoding_info']
        self.skipped_features = save_info['skipped_features']
        self.is_fitted = save_info['is_fitted']
        
        print(f"编码器已从 {filepath} 加载")
        
    def get_encoding_summary(self):
        """
        获取编码摘要信息
        """
        summary = {
            'total_features': len(self.encoding_info),
            'label_encoded': sum(1 for info in self.encoding_info.values() if info['strategy'] == 'LE'),
            'binary_encoded': sum(1 for info in self.encoding_info.values() if info['strategy'] == 'BE'),
            'skipped_features': len(self.skipped_features),
            'encoding_details': {},
            'skipped_details': self.skipped_features
        }
        
        # 添加每个特征的详细信息
        for col, info in self.encoding_info.items():
            summary['encoding_details'][col] = {
                '编码方式': 'Label Encoding' if info['strategy'] == 'LE' else 'Binary Encoding',
                '唯一值数量': info['n_unique_values'],
                '唯一值比例': f"{info['ratio']:.2%}",
                '输出特征数': info['n_output_features']
            }
            
        return summary 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import BinaryEncoder, WOEEncoder
import logging

#logging.basicConfig(
#    level=logging.INFO,
#    filename='autoML.log',
#    format='%(asctime)s - %(levelname)s - %(message)s',
#    encoding='utf-8'
#)

class Encoder:
    """
    编码类，支持多种编码策略
    Parameters:
    ----------
    dataset: DataFrame
        输入数据集
    strategy: str
        编码策略，支持 'LE'(Label Encoding), 'OHE'(One-Hot Encoding), 
        'BE'(Binary Encoding), 'WOE'(Weight of Evidence Encoding)
    target: str
        目标变量名称
    threshold: float, default=0.1
        唯一值比例阈值，当唯一值数量/总样本数 < threshold 时进行编码
    """
    def __init__(self, dataset, strategy='LE', target=None, threshold=0.1):
        self.dataset = dataset
        self.strategy = strategy
        self.target = target
        self.threshold = threshold
        self.encoders = {}
        self.failed_columns = []
        self.skipped_columns = []
        self.successful_columns = []  
        
    def _should_encode(self, X, col):
        """判断是否应该对特征进行编码"""
        try:
            n_unique = X[col].nunique()
            n_samples = len(X)
            ratio = n_unique / n_samples
            
            if ratio >= self.threshold:
                # logging.info(f"特征 {col} 的唯一值比例 ({ratio:.2%}) 超过阈值 ({self.threshold:.2%})，跳过编码")
                self.skipped_columns.append((col, ratio))
                return False
            return True
        except Exception as e:
            # logging.warning(f"检查特征 {col} 时发生错误: {str(e)}")
            self.failed_columns.append((col, 'check_failed'))
            return False
        
    def _safe_encode(self, X, col, encoder_type):
        """安全编码单个特征"""
        try:
            if not self._should_encode(X, col):
                return None
                
            if encoder_type == 'LE':
                # 检查是否为二分类特征
                if X[col].nunique() != 2:
                    # logging.info(f"特征 {col} 不是二分类特征（唯一值数量: {X[col].nunique()}），跳过LE编码")
                    self.skipped_columns.append((col, f"非二分类特征({X[col].nunique()}个唯一值)"))
                    return None
                self.encoders[col] = LabelEncoder()
                encoded = self.encoders[col].fit_transform(X[col].astype(str))
                self.successful_columns.append((col, 'LE'))  # 记录成功编码
                return encoded
            elif encoder_type == 'OHE':
                self.encoders[col] = OneHotEncoder(sparse=False, drop='first')
                encoded = self.encoders[col].fit_transform(X[[col]])
                self.successful_columns.append((col, 'OHE'))  # 记录成功编码
                return pd.DataFrame(encoded, 
                                  columns=[f"{col}_{i}" for i in range(encoded.shape[1])],
                                  index=X.index)
            elif encoder_type == 'BE':
                self.encoders[col] = BinaryEncoder()
                encoded = self.encoders[col].fit_transform(X[[col]])
                self.successful_columns.append((col, 'BE'))  # 记录成功编码
                return encoded
            elif encoder_type == 'WOE':
                if self.target is not None and self.target in X.columns:
                    self.encoders[col] = WOEEncoder()
                    encoded = self.encoders[col].fit_transform(X[[col]], X[self.target])
                    self.successful_columns.append((col, 'WOE'))  # 记录成功编码
                    return encoded
                else:
                    # logging.warning(f"特征 {col} 无法使用WOE编码：缺少目标变量")
                    self.failed_columns.append((col, 'woe_no_target'))
                    return None
        except Exception as e:
            # logging.warning(f"特征 {col} 的 {encoder_type} 编码失败: {str(e)}")
            self.failed_columns.append((col, encoder_type))
            return None
            
    def transform(self):
        """执行编码转换"""
        if self.strategy == '':
            return self.dataset
            
        X = self.dataset.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return X
            
        # 记录原始列名
        original_columns = set(X.columns)
        
        for col in categorical_cols:
            try:
                encoded = self._safe_encode(X, col, self.strategy)
                if encoded is not None:
                    if isinstance(encoded, pd.DataFrame):
                        X = pd.concat([X.drop(col, axis=1), encoded], axis=1)
                    else:
                        X[col] = encoded
            except Exception as e:
                # logging.error(f"处理特征 {col} 时发生错误: {str(e)}")
                self.failed_columns.append((col, self.strategy))
                continue
                
        # 记录处理结果
        #if self.successful_columns:
        #    logging.info(f"成功编码的特征: {self.successful_columns}")
        # if self.failed_columns:
        #     logging.warning(f"以下特征编码失败: {self.failed_columns}")
        # if self.skipped_columns:
        #     logging.info(f"以下特征被跳过编码: {self.skipped_columns}")
            
        # 确保所有原始列都保留在结果中
        missing_columns = original_columns - set(X.columns)
        if missing_columns:
            # logging.warning(f"以下原始列在编码过程中丢失: {missing_columns}")
            for col in missing_columns:
                if col in self.dataset.columns:
                    X[col] = self.dataset[col]
            
        return X 
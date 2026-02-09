import time
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class Classifier:
    """
    Classification task with hyperparameter support
    ----------
    Parameters:
    * dataset: DataFrame
        Input dataset containing features and target variable
    
    * target: str
        Name of target variable
    
    * strategy: str (default='NB')
        Classification algorithm: 
        'NB', 'LogisticRegression', 'SVM', 'KNN',
        'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost',
        'AdaBoost', 'ExtraTrees', 'MLP'
    
    * hyperparameters: dict (optional)
        Model-specific parameters. Examples:
        {
            'C': 1.0,              # For logistic regression and SVM
            'kernel': 'rbf',       # For SVM
            'n_estimators': 100,   # For tree-based models
            'max_depth': 3,        # For tree-based models
            'learning_rate': 0.1,  # For GradientBoosting and tree-based models
        }
    
    * cv_folds: int (default=5)
        Number of folds for cross-validation
    
    * cv_scoring: str (default='accuracy')
        Scoring metric for cross-validation
    """

    def __init__(self, dataset, target, strategy='NB', hyperparameters=None, cv_folds=5, cv_scoring='accuracy'):
        self.dataset = dataset
        self.target = target
        self.strategy = strategy
        self.hyperparams = self._sanitize_hyperparams(hyperparameters or {})
        self.cv_folds = cv_folds
        self.cv_scoring = cv_scoring
        self.le = LabelEncoder()
        self.trained_model = None  # 存储训练好的模型

    @staticmethod
    def _sanitize_hyperparams(params):
        """验证并修正超参数，防止非法值导致模型训练崩溃"""
        sanitized = dict(params)
        # 定义参数最小值约束
        min_constraints = {
            'num_leaves': 2,           # LightGBM: must be > 1
            'max_iter': 10,            # MLP/LogisticRegression: must be >= 1
            'n_estimators': 2,         # Tree models: must be >= 2
            'max_depth': 1,            # must be >= 1 (or None)
            'min_samples_split': 2,    # must be >= 2
            'min_samples_leaf': 1,     # must be >= 1
            'min_child_samples': 1,    # LightGBM: must be >= 1
            'iterations': 2,           # CatBoost: must be >= 2
            'depth': 1,               # CatBoost: must be >= 1
            'border_count': 1,        # CatBoost: must be >= 1
        }
        for param, min_val in min_constraints.items():
            if param in sanitized and sanitized[param] is not None:
                if isinstance(sanitized[param], (int, float)):
                    if sanitized[param] < min_val:
                        sanitized[param] = min_val
        return sanitized

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'target': self.target,
            'hyperparameters': self.hyperparams,
            'cv_folds': self.cv_folds,
            'cv_scoring': self.cv_scoring
        }

    def set_params(self, **params):
        valid_params = self.get_params().keys()
        for k, v in params.items():
            if k == 'hyperparameters':
                self.hyperparams.update(v)
            elif k in valid_params:
                setattr(self, k, v)
            else:
                warnings.warn(f"Ignored invalid parameter: {k}")

    def _get_clean_data(self):
        """准备数据用于交叉验证"""
        X = self.dataset.drop(self.target, axis=1)
        
        # 对仍然是 object 类型的列进行 Label Encoding
        object_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in object_cols:
            try:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            except Exception as e:
                # 如果编码失败，删除该列
                X = X.drop(columns=[col])
        
        # 现在选择数值列（包括刚编码的）
        X = X.select_dtypes(['number']).dropna()
        y = self.dataset[self.target].loc[X.index]
        
        # 编码目标变量 - 确保转换为整数类型
        # 处理可能的category/string类型
        if y.dtype == 'object' or str(y.dtype) == 'category':
            y = self.le.fit_transform(y.astype(str))
        else:
            y = self.le.fit_transform(y)
        
        # 确保y是整数类型（避免nan）
        y = np.asarray(y, dtype=int)
        
        if len(X.columns) < 1 or len(X) < 5:
            raise ValueError("Need >=1 numeric feature and >=5 samples")
        
        # 保存实际使用的特征列
        self.feature_columns = list(X.columns)
            
        return X, y

    def NB_classification(self):
        X, y = self._get_clean_data()
        
        model = GaussianNB()
        # 使用分层交叉验证，保证每个fold中类别分布一致
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def LogisticRegression_classification(self):
        X, y = self._get_clean_data()
        
        model = LogisticRegression(
            C=self.hyperparams.get('C', 1.0),
            solver=self.hyperparams.get('solver', 'liblinear'),
            max_iter=self.hyperparams.get('max_iter', 100),
            random_state=42
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def SVM_classification(self):
        X, y = self._get_clean_data()
        
        model = SVC(
            kernel=self.hyperparams.get('kernel', 'rbf'),
            C=self.hyperparams.get('C', 1.0),
            gamma=self.hyperparams.get('gamma', 'scale'),
            random_state=42,
            probability=True  # 启用概率预测
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def KNN_classification(self):
        X, y = self._get_clean_data()
        
        model = KNeighborsClassifier(
            n_neighbors=self.hyperparams.get('n_neighbors', 5),
            weights=self.hyperparams.get('weights', 'uniform'),
            algorithm=self.hyperparams.get('algorithm', 'auto')
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def RandomForest_classification(self):
        X, y = self._get_clean_data()
        
        model = RandomForestClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', None),
            min_samples_split=self.hyperparams.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
            max_features=self.hyperparams.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=1
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def GradientBoosting_classification(self):
        X, y = self._get_clean_data()
        
        model = GradientBoostingClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            max_depth=self.hyperparams.get('max_depth', 3),
            subsample=self.hyperparams.get('subsample', 1.0),
            min_samples_split=self.hyperparams.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
            random_state=42
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def XGBoost_classification(self):
        X, y = self._get_clean_data()
        
        model = XGBClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', 3),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            subsample=self.hyperparams.get('subsample', 1.0),
            colsample_bytree=self.hyperparams.get('colsample_bytree', 1.0),
            min_child_weight=self.hyperparams.get('min_child_weight', 1),
            gamma=self.hyperparams.get('gamma', 0),
            reg_alpha=self.hyperparams.get('reg_alpha', 0),
            reg_lambda=self.hyperparams.get('reg_lambda', 1),
            random_state=42,
            n_jobs=1,
            verbosity=0
            # 移除early_stopping_rounds，因为cross_val_score不支持验证集
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型（不使用早停）
        model_final = XGBClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', 3),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            subsample=self.hyperparams.get('subsample', 1.0),
            colsample_bytree=self.hyperparams.get('colsample_bytree', 1.0),
            min_child_weight=self.hyperparams.get('min_child_weight', 1),
            gamma=self.hyperparams.get('gamma', 0),
            reg_alpha=self.hyperparams.get('reg_alpha', 0),
            reg_lambda=self.hyperparams.get('reg_lambda', 1),
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        model_final.fit(X, y)
        self.trained_model = model_final
        
        return fitness

    def LightGBM_classification(self):
        X, y = self._get_clean_data()
        
        model = LGBMClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', 3),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            num_leaves=self.hyperparams.get('num_leaves', 31),
            subsample=self.hyperparams.get('subsample', 1.0),
            colsample_bytree=self.hyperparams.get('colsample_bytree', 1.0),
            reg_alpha=self.hyperparams.get('reg_alpha', 0),
            reg_lambda=self.hyperparams.get('reg_lambda', 0),
            min_child_samples=self.hyperparams.get('min_child_samples', 20),
            random_state=42,
            verbose=-1,
            force_col_wise=True  # 避免警告
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def CatBoost_classification(self):
        X, y = self._get_clean_data()
        
        model = CatBoostClassifier(
            iterations=self.hyperparams.get('iterations', 100),
            depth=self.hyperparams.get('depth', 6),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            l2_leaf_reg=self.hyperparams.get('l2_leaf_reg', 3),
            bootstrap_type=self.hyperparams.get('bootstrap_type', 'Bernoulli'),
            border_count=self.hyperparams.get('border_count', 128),
            random_seed=42,
            verbose=False,
            allow_writing_files=False  # 避免写入临时文件
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def AdaBoost_classification(self):
        X, y = self._get_clean_data()
        
        model = AdaBoostClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 50),
            learning_rate=self.hyperparams.get('learning_rate', 1.0),
            algorithm=self.hyperparams.get('algorithm', 'SAMME'),
            random_state=42
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def ExtraTrees_classification(self):
        X, y = self._get_clean_data()
        
        model = ExtraTreesClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', None),
            min_samples_split=self.hyperparams.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
            max_features=self.hyperparams.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=1
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def MLP_classification(self):
        X, y = self._get_clean_data()
        
        model = MLPClassifier(
            hidden_layer_sizes=self.hyperparams.get('hidden_layer_sizes', (100,)),
            activation=self.hyperparams.get('activation', 'relu'),
            solver=self.hyperparams.get('solver', 'adam'),
            alpha=self.hyperparams.get('alpha', 0.0001),
            learning_rate=self.hyperparams.get('learning_rate', 'constant'),
            max_iter=self.hyperparams.get('max_iter', 200),
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def transform(self):
        import logging
        start_time = time.time()
        
        if self.target not in self.dataset.columns:
            raise ValueError("Target variable not found in dataset")
            
        print(f"\n>> {self.strategy} Classification with params: {self.hyperparams}")
        print(f">> Cross-validation folds: {self.cv_folds}")
        
        method_name = f"{self.strategy}_classification"
        if not hasattr(self, method_name):
            raise ValueError(f"Invalid strategy: {self.strategy}")
            
        try:
            fitness = getattr(self, method_name)()
            print(f"Completed in {time.time()-start_time:.2f}s | Fitness (CV mean): {fitness:.4f}")
            # 返回适应度、训练好的模型和实际使用的特征列
            return fitness, self.trained_model, getattr(self, 'feature_columns', None)
        except ValueError as e:
            logging.error(f"Classification ValueError: {str(e)}")
            print(f"Classification failed (ValueError): {str(e)}")
            return -np.inf, None, None  # 返回-inf而不是0.0，与其他地方保持一致
        except Exception as e:
            # 捕获所有其他异常（包括sklearn的评分错误）
            logging.error(f"Classification failed: {type(e).__name__}: {str(e)}")
            print(f"Classification failed ({type(e).__name__}): {str(e)}")
            return -np.inf, None, None
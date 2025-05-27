import time
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
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
        'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'
    
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
        self.hyperparams = hyperparameters or {}
        self.cv_folds = cv_folds
        self.cv_scoring = cv_scoring
        self.le = LabelEncoder()
        self.trained_model = None  # 存储训练好的模型

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
        X = X.select_dtypes(['number']).dropna()
        y = self.dataset[self.target].loc[X.index]
        
        # 编码目标变量
        y = self.le.fit_transform(y)
        
        if len(X.columns) < 1 or len(X) < 5:
            raise ValueError("Need >=1 numeric feature and >=5 samples")
            
        return X, y

    def NB_classification(self):
        X, y = self._get_clean_data()
        
        model = GaussianNB()
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
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
            max_iter=self.hyperparams.get('max_iter', 100)
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
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
            gamma=self.hyperparams.get('gamma', 'scale')
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def KNN_classification(self):
        X, y = self._get_clean_data()
        
        model = KNeighborsClassifier(
            n_neighbors=self.hyperparams.get('n_neighbors', 5)
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
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
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
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
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
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
            random_state=42
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def LightGBM_classification(self):
        X, y = self._get_clean_data()
        
        model = LGBMClassifier(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', 3),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            num_leaves=self.hyperparams.get('num_leaves', 31),
            subsample=self.hyperparams.get('subsample', 1.0),
            random_state=42,
            verbose=-1
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def CatBoost_classification(self):
        X, y = self._get_clean_data()
        
        model = CatBoostClassifier(
            iterations=self.hyperparams.get('iterations', 100),
            depth=self.hyperparams.get('depth', 3),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            l2_leaf_reg=self.hyperparams.get('l2_leaf_reg', 3),
            bootstrap_type=self.hyperparams.get('bootstrap_type', 'Bernoulli'),
            random_seed=42,
            verbose=False
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def transform(self):
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
            return fitness, self.trained_model  # 返回适应度（交叉验证平均分数）和训练好的模型
        except ValueError as e:
            print(f"Classification failed: {str(e)}")
            return 0.0, None
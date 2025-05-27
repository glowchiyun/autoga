import time
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

class Regressor:
    """
    Regression task with hyperparameter support
    ----------
    Parameters:
    * dataset: DataFrame
        Input dataset containing features and target variable
    
    * target: str
        Name of target variable
    
    * strategy: str (default='LASSO')
        Regression algorithm: 
        'OLS', 'LASSO', 'Ridge', 'ElasticNet', 'SVR',
        'RandomForest', 'GradientBoosting'
    
    * hyperparameters: dict (optional)
        Model-specific parameters. Examples:
        {
            'alpha': 0.01,          # For linear models
            'l1_ratio': 0.5,        # For ElasticNet
            'C': 1.0,              # For SVR
            'gamma': 'scale',       # For SVR
            'n_estimators': 100,    # For tree-based models
            'max_depth': 3,        # For tree-based models
            'learning_rate': 0.1,  # For GradientBoosting
            'kernel': 'rbf'        # For SVR
        }
    
    * cv_folds: int (default=5)
        Number of folds for cross-validation
    * cv_scoring: str (default='neg_mean_squared_error')
        Scoring metric for cross-validation
    """

    def __init__(self, dataset, target, strategy='LASSO', 
                 hyperparameters=None, cv_folds=5, cv_scoring='neg_mean_squared_error'):
        self.dataset = dataset
        self.target = target
        self.strategy = strategy
        self.hyperparams = hyperparameters or {}
        self.cv_folds = cv_folds
        self.trained_model = None  
        self.cv_scoring = cv_scoring

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'target': self.target,
            'hyperparameters': self.hyperparams,
            'cv_folds': self.cv_folds
        }

    def set_params(self,**params):
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
        
        if len(X.columns) < 1 or len(X) < 5:
            raise ValueError("Need >=1 numeric feature and >=5 samples")
            
        return X, y

    def OLS_regression(self):
      
        X, y = self._get_clean_data()
        model = LinearRegression()
        cv_scores = cross_val_score(estimator=model,X=X,y=y,cv=self.cv_folds,scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        model.fit(X, y)
        self.trained_model = model
            
        return fitness    

    def LASSO_regression(self):
        X, y = self._get_clean_data()
        
        model = Lasso(
            alpha=self.hyperparams.get('alpha', 1.0),
            max_iter=self.hyperparams.get('max_iter', 1000),
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def Ridge_regression(self):
        X, y = self._get_clean_data()
        
        model = Ridge(
            alpha=self.hyperparams.get('alpha', 1.0),
            max_iter=self.hyperparams.get('max_iter', None),
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def ElasticNet_regression(self):
        X, y = self._get_clean_data()
        
        model = ElasticNet(
            alpha=self.hyperparams.get('alpha', 1.0),
            l1_ratio=self.hyperparams.get('l1_ratio', 0.5),
            max_iter=self.hyperparams.get('max_iter', 1000),
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def SVR_regression(self):
        X, y = self._get_clean_data()
        
        model = SVR(
            kernel=self.hyperparams.get('kernel', 'rbf'),
            C=self.hyperparams.get('C', 1.0),
            gamma=self.hyperparams.get('gamma', 'scale'),
            epsilon=self.hyperparams.get('epsilon', 0.1)
        )
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        # 在完整数据集上训练模型
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def RandomForest_regression(self):
        X, y = self._get_clean_data()
        
        model = RandomForestRegressor(
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

    def GradientBoosting_regression(self):
        X, y = self._get_clean_data()
        
        model = GradientBoostingRegressor(
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

    def transform(self):
        start_time = time.time()
        
        if self.target not in self.dataset.columns:
            raise ValueError("Target variable not found in dataset")
            
        print(f"\n>> {self.strategy} Regression with params: {self.hyperparams}")
        print(f">> Cross-validation folds: {self.cv_folds}")
        
        method_name = f"{self.strategy}_regression"
        if not hasattr(self, method_name):
            raise ValueError(f"Invalid strategy: {self.strategy}")
            
        try:
            fitness = getattr(self, method_name)()
            print(f"Completed in {time.time()-start_time:.2f}s | Fitness (CV mean): {fitness:.4f}")
            return fitness, self.trained_model  # 返回适应度（交叉验证平均分数）和训练好的模型
        except ValueError as e:
            print(f"Regression failed: {str(e)}")
            return 0.0, None


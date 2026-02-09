import time
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
        'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost',
        'AdaBoost', 'ExtraTrees', 'MLP'
    
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
        self.hyperparams = self._sanitize_hyperparams(hyperparameters or {})
        self.cv_folds = cv_folds
        self.trained_model = None  
        self.cv_scoring = cv_scoring

    @staticmethod
    def _sanitize_hyperparams(params):
        """验证并修正超参数，防止非法值导致模型训练崩溃"""
        sanitized = dict(params)
        min_constraints = {
            'num_leaves': 2,
            'max_iter': 10,
            'n_estimators': 2,
            'max_depth': 1,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_child_samples': 1,
            'iterations': 2,
            'depth': 1,
            'border_count': 1,
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
        from sklearn.preprocessing import LabelEncoder
        
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
        
        if len(X.columns) < 1 or len(X) < 5:
            raise ValueError("Need >=1 numeric feature and >=5 samples")
        
        # 保存实际使用的特征列
        self.feature_columns = list(X.columns)
            
        return X, y

    def _get_cv(self):
        """获取交叉验证对象，使用shuffle提高稳定性"""
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

    def OLS_regression(self):
        X, y = self._get_clean_data()
        model = LinearRegression()
        cv = self._get_cv()
        cv_scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        model.fit(X, y)
        self.trained_model = model
            
        return fitness    

    def LASSO_regression(self):
        X, y = self._get_clean_data()
        
        model = Lasso(
            alpha=self.hyperparams.get('alpha', 1.0),
            max_iter=self.hyperparams.get('max_iter', 1000),
            random_state=42
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def Ridge_regression(self):
        X, y = self._get_clean_data()
        
        model = Ridge(
            alpha=self.hyperparams.get('alpha', 1.0),
            max_iter=self.hyperparams.get('max_iter', None),
            random_state=42
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def ElasticNet_regression(self):
        X, y = self._get_clean_data()
        
        model = ElasticNet(
            alpha=self.hyperparams.get('alpha', 1.0),
            l1_ratio=self.hyperparams.get('l1_ratio', 0.5),
            max_iter=self.hyperparams.get('max_iter', 1000),
            random_state=42
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
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
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def RandomForest_regression(self):
        X, y = self._get_clean_data()
        
        model = RandomForestRegressor(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', None),
            min_samples_split=self.hyperparams.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
            max_features=self.hyperparams.get('max_features', 1.0),
            random_state=42,
            n_jobs=1
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
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
            min_samples_split=self.hyperparams.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
            random_state=42
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def XGBoost_regression(self):
        X, y = self._get_clean_data()
        
        model = XGBRegressor(
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
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def LightGBM_regression(self):
        X, y = self._get_clean_data()
        
        model = LGBMRegressor(
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
            force_col_wise=True
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def CatBoost_regression(self):
        X, y = self._get_clean_data()
        
        model = CatBoostRegressor(
            iterations=self.hyperparams.get('iterations', 100),
            depth=self.hyperparams.get('depth', 6),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            l2_leaf_reg=self.hyperparams.get('l2_leaf_reg', 3),
            bootstrap_type=self.hyperparams.get('bootstrap_type', 'Bayesian'),
            border_count=self.hyperparams.get('border_count', 128),
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def AdaBoost_regression(self):
        X, y = self._get_clean_data()
        
        model = AdaBoostRegressor(
            n_estimators=self.hyperparams.get('n_estimators', 50),
            learning_rate=self.hyperparams.get('learning_rate', 1.0),
            loss=self.hyperparams.get('loss', 'linear'),
            random_state=42
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def ExtraTrees_regression(self):
        X, y = self._get_clean_data()
        
        model = ExtraTreesRegressor(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            max_depth=self.hyperparams.get('max_depth', None),
            min_samples_split=self.hyperparams.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparams.get('min_samples_leaf', 1),
            max_features=self.hyperparams.get('max_features', 1.0),
            random_state=42,
            n_jobs=1
        )
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def MLP_regression(self):
        X, y = self._get_clean_data()
        
        model = MLPRegressor(
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
        cv = self._get_cv()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.cv_scoring)
        fitness = np.mean(cv_scores)
        
        model.fit(X, y)
        self.trained_model = model
        
        return fitness

    def transform(self):
        import logging
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
            # 返回适应度、训练好的模型和实际使用的特征列
            return fitness, self.trained_model, getattr(self, 'feature_columns', None)
        except ValueError as e:
            logging.error(f"Regression ValueError: {str(e)}")
            print(f"Regression failed (ValueError): {str(e)}")
            return -np.inf, None, None
        except Exception as e:
            # 捕获所有其他异常（包括sklearn的评分错误）
            logging.error(f"Regression failed: {type(e).__name__}: {str(e)}")
            print(f"Regression failed ({type(e).__name__}): {str(e)}")
            return -np.inf, None, None


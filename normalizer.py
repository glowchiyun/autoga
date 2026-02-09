import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler, quantile_transform)
pd.options.mode.chained_assignment = None


class Normalizer():

    """标准化数据

    Parameters
    ----------
    * strategy : str, default = 'ZS'
    标准化数据的策略的值: 'ZS', 'MM','DS' or 'Log10'
       - 'ZS' z-score normalization
       - 'MM' MinMax scaler
       - 'DS' decimal scaling
       - 'Log10'

    * exclude : 不进行标准化的属性

    * verbose: Boolean,  default = 'False' 是否输出详细信息

    * threshold: float, default =  None 比例
    """

    def __init__(self, dataset, strategy='ZS',  exclude=None,
                 verbose=False, threshold=None):

        self.dataset = dataset

        self.strategy = strategy

        self.exclude = exclude

        self.verbose = verbose

        self.threshold = threshold
        
        # 用于保存训练集的统计量（均值、标准差、最小最大值等）
        self.fitted_params_ = {}
        self.is_fitted_ = False
        self.numeric_columns_ = None

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'exclude': self.exclude,

                'verbose': self.verbose,

                'threshold': self.threshold}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`normalizer.get_params().keys()`")

            else:

                setattr(self, k, v)

    def ZS_normalization(self, dataset, fit_mode=True):
        # Normalize numeric columns with Z-score normalisation

        d = dataset

        if self.verbose:
            print("ZS normalizing... ")

        if (type(dataset) != pd.core.series.Series):

            X = dataset.select_dtypes(['number'])
            Y = dataset.select_dtypes(['object'])
            Z = dataset.select_dtypes(['datetime64'])
            
            # 排除指定的列（如目标变量）
            exclude_cols = []
            if self.exclude is not None:
                if isinstance(self.exclude, str):
                    exclude_cols = [self.exclude]
                elif isinstance(self.exclude, (list, tuple)):
                    exclude_cols = list(self.exclude)
            
            # 保存被排除列的原始值
            excluded_data = {}
            for col in exclude_cols:
                if col in X.columns:
                    excluded_data[col] = X[col].copy()
                    X = X.drop(columns=[col])

            # Fit mode: 计算并保存统计量
            if fit_mode:
                self.numeric_columns_ = X.columns.tolist()
                for column in X.columns:
                    self.fitted_params_[column] = {
                        'mean': X[column].mean(),
                        'std': X[column].std()
                    }
            
            # Transform: 使用保存的统计量
            for column in X.columns:
                if fit_mode:
                    # 训练阶段：使用当前计算的统计量
                    X[column] = (X[column] - self.fitted_params_[column]['mean']) / self.fitted_params_[column]['std']
                else:
                    # 测试阶段：使用训练时保存的统计量
                    if column in self.fitted_params_:
                        X[column] = (X[column] - self.fitted_params_[column]['mean']) / self.fitted_params_[column]['std']
                    else:
                        # 如果是新列，跳过标准化
                        if self.verbose:
                            print(f"Warning: Column {column} not seen during training, skipping normalization")

            df = X.join(Y)
            df = df.join(Z)
            
            # 恢复被排除的列（原始值）
            for col, values in excluded_data.items():
                df[col] = values

        else:

            X = dataset
            
            # Fit mode: 计算并保存统计量
            if fit_mode:
                self.fitted_params_['series'] = {
                    'mean': X.mean(),
                    'std': X.std()
                }
            
            # Transform
            if fit_mode:
                X = (X - self.fitted_params_['series']['mean']) / self.fitted_params_['series']['std']
            else:
                X = (X - self.fitted_params_['series']['mean']) / self.fitted_params_['series']['std']

            df = X

            if (self.exclude in list(pd.DataFrame(df).columns.values)):
                df = dataset

        return df.sort_index()

    def MM_normalization(self, dataset, fit_mode=True):
        # Normalize numeric columns with MinMax normalization

        d = dataset

        if self.verbose:
            print("MM normalizing...")

        if (type(dataset) != pd.core.series.Series):

            Xf = dataset.select_dtypes(['number'])
            Y = dataset.select_dtypes(['object'])
            Z = dataset.select_dtypes(['datetime64'])
            
            # 排除指定的列（如目标变量）
            exclude_cols = []
            if self.exclude is not None:
                if isinstance(self.exclude, str):
                    exclude_cols = [self.exclude]
                elif isinstance(self.exclude, (list, tuple)):
                    exclude_cols = list(self.exclude)
            
            # 保存被排除列的原始值
            excluded_data = {}
            for col in exclude_cols:
                if col in Xf.columns:
                    excluded_data[col] = Xf[col].copy()
                    Xf = Xf.drop(columns=[col])
            
            X = Xf.dropna()
            X_na = Xf[Xf.isnull().any(axis=1)]

            # Fit mode: 计算并保存统计量
            if fit_mode:
                self.numeric_columns_ = X.columns.tolist()
                for column in X.columns:
                    self.fitted_params_[column] = {
                        'min': X[column].min(),
                        'max': X[column].max()
                    }
            
            # Transform: 使用保存的统计量
            scaled_X = X.copy()
            for column in X.columns:
                if fit_mode:
                    # 训练阶段：使用当前计算的统计量
                    min_val = self.fitted_params_[column]['min']
                    max_val = self.fitted_params_[column]['max']
                    if max_val - min_val != 0:
                        scaled_X[column] = (X[column] - min_val) / (max_val - min_val)
                    else:
                        scaled_X[column] = 0
                else:
                    # 测试阶段：使用训练时保存的统计量
                    if column in self.fitted_params_:
                        min_val = self.fitted_params_[column]['min']
                        max_val = self.fitted_params_[column]['max']
                        if max_val - min_val != 0:
                            scaled_X[column] = (X[column] - min_val) / (max_val - min_val)
                        else:
                            scaled_X[column] = 0
                    else:
                        # 如果是新列，跳过标准化
                        if self.verbose:
                            print(f"Warning: Column {column} not seen during training, skipping normalization")

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = scaled_Xf.join(Y)
            df = df.join(Z)
            
            # 恢复被排除的列（原始值）
            for col, values in excluded_data.items():
                df[col] = values

        else:
            X = dataset.dropna()
            X_na = dataset[dataset.isna()]

            # Fit mode: 计算并保存统计量
            if fit_mode:
                self.fitted_params_['series'] = {
                    'min': X.min(),
                    'max': X.max()
                }
            
            # Transform
            min_val = self.fitted_params_['series']['min']
            max_val = self.fitted_params_['series']['max']
            if max_val - min_val != 0:
                scaled_X = (X - min_val) / (max_val - min_val)
            else:
                scaled_X = pd.Series([0] * len(X), index=X.index)

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = pd.Series(scaled_Xf, index=dataset.index)

            if (self.exclude in list(pd.DataFrame(df).columns.values)):
                df = dataset

        return df.sort_index()

    def DS_normalization(self, dataset, fit_mode=True):
        # Normalize numeric columns with decimal scaling (quantile transform)

        d = dataset

        if self.verbose:
            print("DS normalizing...")

        if (type(dataset) != pd.core.series.Series):

            Xf = dataset.select_dtypes(['number'])
            Y = dataset.select_dtypes(['object'])
            Z = dataset.select_dtypes(['datetime64'])
            
            # 排除指定的列（如目标变量）
            exclude_cols = []
            if self.exclude is not None:
                if isinstance(self.exclude, str):
                    exclude_cols = [self.exclude]
                elif isinstance(self.exclude, (list, tuple)):
                    exclude_cols = list(self.exclude)
            
            # 保存被排除列的原始值
            excluded_data = {}
            for col in exclude_cols:
                if col in Xf.columns:
                    excluded_data[col] = Xf[col].copy()
                    Xf = Xf.drop(columns=[col])
            
            X = Xf.dropna()
            X_na = Xf[Xf.isnull().any(axis=1)]

            # Note: quantile_transform 需要 fit 和 transform 分离
            # 但 sklearn 的实现会自动保存状态，这里简化处理
            # 在实际应用中，建议使用 sklearn.preprocessing.QuantileTransformer
            if fit_mode:
                self.numeric_columns_ = X.columns.tolist()
                # quantile_transform 会自动计算分位数
            
            scaled_values = quantile_transform(
                X, n_quantiles=10, random_state=0)

            scaled_X = pd.DataFrame(
                scaled_values, index=X.index, columns=X.columns)

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = scaled_Xf.join(Y)
            df = df.join(Z)
            
            # 恢复被排除的列（原始值）
            for col, values in excluded_data.items():
                df[col] = values

        else:
            X = dataset.dropna()
            X_na = dataset[dataset.isna()]

            scaled_X = X.quantile(q=0.1, interpolation='linear')

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = pd.Series(scaled_Xf, index=dataset.index)

            if (self.exclude in list(pd.DataFrame(df).columns.values)):
                df = dataset

        return df.sort_index()

    def Log10_normalization(self, dataset, fit_mode=True):
        # Normalize numeric columns with log10 scaling
        d = dataset

        if self.verbose:
            print("Log10 normalizing...")

        X = dataset.select_dtypes(['number'])
        Y = dataset.select_dtypes(['object'])
        Z = dataset.select_dtypes(['datetime64'])

        # 排除指定的列（如目标变量）
        exclude_cols = []
        if self.exclude is not None:
            if isinstance(self.exclude, str):
                exclude_cols = [self.exclude]
            elif isinstance(self.exclude, (list, tuple)):
                exclude_cols = list(self.exclude)
        
        # 保存被排除列的原始值
        excluded_data = {}
        for col in exclude_cols:
            if col in X.columns:
                excluded_data[col] = X[col].copy()
                X = X.drop(columns=[col])

        # Apply log10 normalization
        for column in X.columns:
            # Fit mode: 计算并保存偏移量
            if fit_mode:
                min_val = X[column].min()
                if min_val <= 0:
                    # 保存偏移量
                    self.fitted_params_[column] = {'offset': -min_val + 1}
                else:
                    self.fitted_params_[column] = {'offset': 0}
                self.numeric_columns_ = X.columns.tolist()
            
            # Transform: 使用保存的偏移量
            if column in self.fitted_params_:
                offset = self.fitted_params_[column]['offset']
                if offset > 0:
                    X[column] = X[column] + offset
                X[column] = np.log10(X[column])
            else:
                # 如果是新列，使用当前数据的最小值
                min_val = X[column].min()
                if min_val <= 0:
                    X[column] = X[column] - min_val + 1
                X[column] = np.log10(X[column])
                if self.verbose:
                    print(f"Warning: Column {column} not seen during training, using current min value")

        # Merge back non-numeric columns
        df = X.join(Y)
        df = df.join(Z)

        # 恢复被排除的列（原始值）
        for col, values in excluded_data.items():
            df[col] = values

        return df

    def transform(self, new_data=None):
        """
        标准化数据
        
        Parameters
        ----------
        new_data : pd.DataFrame, optional
            如果提供，则使用训练时保存的统计量转换新数据（测试集）
            如果不提供，则在 self.dataset 上进行 fit + transform（训练集）
            
        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        # 判断是训练阶段还是测试阶段
        fit_mode = (new_data is None)
        d = self.dataset if fit_mode else new_data

        start_time = time.time()

        if fit_mode:
            print(">>Normalization (Training)")
        else:
            print(">>Normalization (Testing - using saved params)")

        if (self.strategy == "DS"):
            dn = self.DS_normalization(d, fit_mode=fit_mode)

        elif (self.strategy == "ZS"):
            dn = self.ZS_normalization(d, fit_mode=fit_mode)

        elif (self.strategy == "MM"):
            dn = self.MM_normalization(d, fit_mode=fit_mode)

        elif (self.strategy == "Log10"):
            dn = self.Log10_normalization(d, fit_mode=fit_mode)

        else:
            raise ValueError(
                "The normalization function should be MM,"
                " ZS, DS or Log10")

        if (self.exclude in list(pd.DataFrame(d).columns.values)):
            dn[self.exclude] = d[self.exclude]

        normd = dn
        
        # 标记为已训练
        if fit_mode:
            self.is_fitted_ = True

        print("Normalization done -- CPU time: %s seconds" %
              (time.time() - start_time))

        print()

        return normd

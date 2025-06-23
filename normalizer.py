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

    def ZS_normalization(self, dataset):
        # Normalize numeric columns with Z-score normalisation

        d = dataset

        if self.verbose:

            print("ZS normalizing... ")

        if (type(dataset) != pd.core.series.Series):

            X = dataset.select_dtypes(['number'])

            Y = dataset.select_dtypes(['object'])

            Z = dataset.select_dtypes(['datetime64'])

            for column in X.columns:

                X[column] -= X[column].mean()

                X[column] /= X[column].std()

            df = X.join(Y)

            df = df.join(Z)

            if (self.exclude in list(df.columns.values)):

                df[str(self.exclude)] = d[str(self.exclude)].values

        else:

            X = dataset

            X -= X.mean()

            X /= X.std()

            df = X

            if (self.exclude in list(pd.DataFrame(df).columns.values)):

                df = dataset

        return df.sort_index()

    def MM_normalization(self, dataset):
        # Normalize numeric columns with MinMax normalization

        d = dataset

        if self.verbose:

            print("MM normalizing...")

        if (type(dataset) != pd.core.series.Series):

            Xf = dataset.select_dtypes(['number'])

            X = Xf.dropna()

            X_na = Xf[Xf.isnull().any(axis=1)]

            Y = dataset.select_dtypes(['object'])

            Z = dataset.select_dtypes(['datetime64'])

            scaled_values = MinMaxScaler().fit_transform(X)

            scaled_X = pd.DataFrame(
                scaled_values, index=X.index, columns=X.columns)

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = scaled_Xf.join(Y)

            df = df.join(Z)

            if (self.exclude in list(df.columns.values)):

                df[str(self.exclude)] = d[str(self.exclude)].values

            # print('Exclude variable is not in the input dataset')
        else:
            # elif (sum([type(x)=='number' for x in dataset])/len(dataset)==1):

            X = dataset.dropna()

            X_na = dataset[dataset.isna()]

            scaled_X = MinMaxScaler().fit_transform(X.values)

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = pd.Series(scaled_Xf, index=X.index, columns=X.columns)

            if (self.exclude in list(pd.DataFrame(df).columns.values)):

                df = dataset
            # else:
            #       print('Exclude variable is not in the input dataset')

        return df.sort_index()

    def DS_normalization(self, dataset):
        # Normalize numeric columns with MinMax normalization

        d = dataset

        if self.verbose:

            print("DS normalizing...")

        if (type(dataset) != pd.core.series.Series):

            Xf = dataset.select_dtypes(['number'])

            X = Xf.dropna()

            X_na = Xf[Xf.isnull().any(axis=1)]

            Y = dataset.select_dtypes(['object'])

            Z = dataset.select_dtypes(['datetime64'])

            scaled_values = quantile_transform(
                X, n_quantiles=10, random_state=0)

            scaled_X = pd.DataFrame(
                scaled_values, index=X.index, columns=X.columns)

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = scaled_Xf.join(Y)

            df = df.join(Z)

            if (self.exclude in list(df.columns.values)):

                df[str(self.exclude)] = d[str(self.exclude)].values

        else:  # (sum([type(x)=='number' for x in dataset])/len(dataset)==1):

            X = dataset.dropna()

            X_na = dataset[dataset.isna()]

            scaled_X = X.quantile(q=0.1, interpolation='linear')

            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()

            df = pd.Series(scaled_Xf, index=X.index, columns=X.columns)

            if (self.exclude in list(pd.DataFrame(df).columns.values)):

                df = dataset

        return df.sort_index()

    def Log10_normalization(self, dataset):
        # Normalize numeric columns with log10 scaling
        d = dataset

        if self.verbose:
            print("Log10 normalizing...")

        X = dataset.select_dtypes(['number'])
        Y = dataset.select_dtypes(['object'])
        Z = dataset.select_dtypes(['datetime64'])

        # Handle exclude column
        if self.exclude in X.columns:
            X = X.drop(columns=[self.exclude])

        # Apply log10 normalization
        for column in X.columns:
            # 确保数据为正数，避免 log10 的无效值
            min_val = X[column].min()
            if min_val <= 0:
                # 如果最小值小于等于0，将所有值平移使其为正
                X[column] = X[column] - min_val + 1
            X[column] = np.log10(X[column])

        # Merge back non-numeric columns
        df = X.join(Y)
        df = df.join(Z)

        # Handle exclude column if it was originally in the dataset
        if self.exclude in d.columns:
            df[self.exclude] = d[self.exclude].values

        return df

    def transform(self):

        # normd=dict.fromkeys(['train','test', 'target'])
        normd = self.dataset

        start_time = time.time()

        print(">>Normalization ")

        d = self.dataset

        if (self.strategy == "DS"):

            dn = self.DS_normalization(d)

        elif (self.strategy == "ZS"):

            dn = self.ZS_normalization(d)

        elif (self.strategy == "MM"):

            dn = self.MM_normalization(d)

        elif (self.strategy == "Log10"):

            dn = self.Log10_normalization(d)

        else:

            raise ValueError(
                "The normalization function should be MM,"
                " ZS, DS or Log10")

        if (self.exclude in list(pd.DataFrame(d).columns.values)):

            dn[self.exclude] = d[self.exclude]

            normd = dn


        print("Normalization done -- CPU time: %s seconds" %
              (time.time() - start_time))

        print()

        return normd

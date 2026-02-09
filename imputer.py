import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class Imputer():
    """
    替换或者删除或对数据进行插补
    Parameters
    ----------
    * dataset: 输入的dataframe
    * strategy: str, default = 'DROP'
        The choice for the feature selection strategy:
            - 'EM': 仅适用于数值变量；基于期望最大化的插补
            - 'MICE': 仅适用于随机缺失（MAR）的数值变量；链式方程的多变量插补
            - 'KNN', 仅适用于数值变量；使用k-最近邻（k=4）插补，该方法使用两行都有观测数据的特征上的平均平方差来加权样本
            - 'RAND', 'MF':适用于数值和分类变量；用变量域中随机选择的值或变量域中最常见的值来替换缺失值
            - 'MEAN', 'MEDIAN': 仅适用于数值变量；用数值变量的平均值或中位数来替换缺失值
                或者 'DROP' 删除至少有一个缺失值的行
    * verbose: Boolean,  default = 'False'
    * threshold: float, default =  None
    * exclude: str, default = 'None'需要从插补中排除的变量名称。
    """

    def __init__(self, dataset, strategy='DROP', verbose=False,
                 exclude=None, threshold=None):

        self.dataset = dataset

        self.strategy = strategy

        self.verbose = verbose

        self.threshold = threshold

        self.exclude = exclude  # to implement
        
        # 用于保存训练集的填充统计量（均值、中位数、众数等）
        self.fitted_params_ = {}
        self.is_fitted_ = False

    def get_params(self, deep=True):

        return {'strategy': self.strategy,
                'verbose': self.verbose,
                'exclude': self.exclude,
                'threshold': self.threshold}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`imputer.get_params().keys()`")

            else:

                setattr(self, k, v)

    # Handling Missing values

    def mean_imputation(self, dataset, fit_mode=True):
        # for numerical data
        # replace missing numerical values by the mean of
        # the corresponding variable

        df = dataset.copy()

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number']).copy()

            for i in X.columns:
                # Fit mode: 计算并保存均值
                if fit_mode:
                    mean_val = X[i].mean()
                    self.fitted_params_[i] = {'mean': mean_val}
                    X[i] = X[i].fillna(mean_val)
                else:
                    # Test mode: 使用保存的均值
                    if i in self.fitted_params_ and 'mean' in self.fitted_params_[i]:
                        X[i] = X[i].fillna(self.fitted_params_[i]['mean'])
                    else:
                        # 如果训练时没有这个特征，使用当前均值
                        X[i] = X[i].fillna(X[i].mean())

            Z = dataset.select_dtypes(exclude=['number'])

            df = pd.concat([X, Z], axis=1)

        else:

            pass

        return df

    def median_imputation(self, dataset, fit_mode=True):
        # only for numerical data
        # replace missing numerical values by the median
        # of the corresponding variable

        df = dataset.copy()

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number']).copy()

            for i in X.columns:
                # Fit mode: 计算并保存中位数
                if fit_mode:
                    median_val = X[i].median()
                    self.fitted_params_[i] = {'median': median_val}
                    X[i] = X[i].fillna(median_val)
                else:
                    # Test mode: 使用保存的中位数
                    if i in self.fitted_params_ and 'median' in self.fitted_params_[i]:
                        X[i] = X[i].fillna(self.fitted_params_[i]['median'])
                    else:
                        # 如果训练时没有这个特征，使用当前中位数
                        X[i] = X[i].fillna(X[i].median())

            Z = dataset.select_dtypes(include=['object'])

            df = pd.concat([X, Z], axis=1)

        else:

            pass

        return df

    def NaN_drop(self, dataset):
        # for both categorical and numerical data
        # drop observations with missing values

        print("Dataset size reduced from", len(
            dataset), "to", len(dataset.dropna()))

        return dataset.dropna()

    def MF_most_frequent_imputation(self, dataset, fit_mode=True):
        # for both categorical and numerical data
        # replace missing values by the most frequent value
        # of the corresponding variable

        df = dataset.copy()
        
        for i in df.columns:
            # Fit mode: 计算并保存众数
            if fit_mode:
                if not df[i].isnull().all():  # 确保列不是全部缺失
                    mfv = df[i].value_counts().idxmax()
                    self.fitted_params_[i] = {'most_frequent': mfv}
                    df[i] = df[i].fillna(mfv)
                    
                    if self.verbose:
                        print("Most frequent value for ", i, "is:", mfv)
            else:
                # Test mode: 使用保存的众数
                if i in self.fitted_params_ and 'most_frequent' in self.fitted_params_[i]:
                    mfv = self.fitted_params_[i]['most_frequent']
                    df[i] = df[i].fillna(mfv)
                    
                    if self.verbose:
                        print("Using saved most frequent value for ", i, ":", mfv)
                else:
                    # 如果训练时没有这个特征，使用当前众数
                    if not df[i].isnull().all():
                        mfv = df[i].value_counts().idxmax()
                        df[i] = df[i].fillna(mfv)

        return df

    def NaN_random_replace(self, dataset):
        # for both categorical and numerical data
        # replace missing data with a random observation with data
        df = dataset.copy()
        
        # 分别处理数值列和非数值列
        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        
        # 对数值列用随机数填充
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if df[col].isna().any():
                    df.loc[df[col].isna(), col] = np.random.randn(df[col].isna().sum())
        
        # 对非数值列用随机已有值填充
        if len(non_numeric_cols) > 0:
            for col in non_numeric_cols:
                if df[col].isna().any():
                    non_na_values = df[col].dropna().values
                    if len(non_na_values) > 0:
                        df.loc[df[col].isna(), col] = np.random.choice(non_na_values, df[col].isna().sum())

        return df

    def KNN_imputation(self, dataset, k=4):
        # only for numerical values
        # Nearest neighbor imputations which weights samples
        # using the mean squared difference on features for which two
        # rows both have observed data.

        from fancyimpute import KNN

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number'])

            for i in X.columns:

                X[i] = KNN(k=k, verbose=False).fit_transform(X)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def MICE_imputation(self, dataset):
        # only for numerical values
        # Multivariate Imputation by Chained Equations only suitable
        # for Missing At Random (MAR),
        # which means that the probability that a value is missing
        # depends only on observed values and not on unobserved values

        import impyute as imp

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = imp.mice(dataset.select_dtypes(['number']).iloc[:, :].values)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def EM_imputation(self, dataset):
        # only for numerical values
        # imputes given data using expectation maximization.
        # E-step: Calculates the expected complete data log
        # likelihood ratio.

        import impyute as imp

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = imp.em(dataset.select_dtypes(['number']).iloc[:, :].values)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(

                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def transform(self, new_data=None):
        """
        填充缺失值
        
        Parameters
        ----------
        new_data : pd.DataFrame, optional
            如果提供，则使用训练时保存的统计量填充新数据（测试集）
            如果不提供，则在 self.dataset 上进行 fit + transform（训练集）
            
        Returns
        -------
        pd.DataFrame
            填充后的数据
        """
        # 判断是训练阶段还是测试阶段
        fit_mode = (new_data is None)
        d = self.dataset if fit_mode else new_data
        
        start_time = time.time()
        if fit_mode:
            print(">>Imputation (Training)")
        else:
            print(">>Imputation (Testing - using saved params)")
        
        impd = d
        dn = d
        total_missing_before = d.isnull().sum().sum()
        Num_missing_before = d.select_dtypes(
        include=['number']).isnull().sum().sum()
        NNum_missing_before = d.select_dtypes(
        exclude=['number']).isnull().sum().sum()
        print("Before imputation:")
        if total_missing_before == 0:
            print("No missing values in the given data")
        else:    
            print("Total", total_missing_before, "missing values in",
                         d.columns[d.isnull().any()].tolist())
        if Num_missing_before > 0:
                        print("-", Num_missing_before,
                              "numerical missing values in",
                              d.select_dtypes(['number']).
                              columns[d.select_dtypes(['number']).
                                      isnull().any()].tolist())
        if NNum_missing_before > 0:

                        print("-", NNum_missing_before,
                              "non-numerical missing values in",
                              d.select_dtypes(['object']).
                              columns[d.select_dtypes(['object']).
                                      isnull().any()].tolist())
        if (self.strategy == "EM"):
            dn = self.EM_imputation(d)
        elif (self.strategy == "MICE"):
            dn = self.MICE_imputation(d)
        elif (self.strategy == "KNN"):
            dn = self.KNN_imputation(d)
        elif (self.strategy == "RAND"):
            dn = self.NaN_random_replace(d)
        elif (self.strategy == "MF"):
            dn = self.MF_most_frequent_imputation(d, fit_mode=fit_mode)
        elif (self.strategy == "MEAN"):
            dn = self.mean_imputation(d, fit_mode=fit_mode)
        elif (self.strategy == "MEDIAN"):
            dn = self.median_imputation(d, fit_mode=fit_mode)
        elif (self.strategy == "DROP"):
            dn = self.NaN_drop(d)
        else:
            raise ValueError("Strategy invalid. Please "
                         "choose between "
                         "'EM', 'MICE', 'KNN', 'RAND', 'MF', "
                         "'MEAN', 'MEDIAN', or 'DROP'")
        impd = dn
        if self.exclude in list(pd.DataFrame(d).columns.values) and self.strategy != "DROP":

            dn[self.exclude] = d[self.exclude]

            impd = dn
        
        print("After imputation:")
        print("Total", impd.isnull().sum().sum(), "missing values")
        print("-", impd.select_dtypes(include=['number']).isnull().sum().sum(),"numerical missing values")
        print("-", impd.select_dtypes(exclude=['number']).isnull().sum().sum(),"non-numerical missing values")
        print("Imputation done -- CPU time: %s seconds" %
              (time.time() - start_time))
        print()
        
        # 标记为已训练
        if fit_mode:
            self.is_fitted_ = True
        
        return impd

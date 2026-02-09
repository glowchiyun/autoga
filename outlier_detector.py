import warnings
import time
import numpy as np


class Outlier_detector():
    """
    识别并移除离群点

    Parameters
    ----------
    * dataset:  dataframe输入文件
    * threshold: float, default = '0.3' 对于任何在行中的异常值，
                或者在多变量情况下，如果一个值在[0,1]范围内，
                如果一行中的异常值数量超过属性集的一半，
                则该行被视为异常行并被移除。
                例如，设定阈值（threshold）为0.5，
                如果一行中一半以上的属性值是异常值，
                那么这行数据就被认为是异常的并被删除

    * strategy: str, default = 'ZSB'
                对于数值数据，使用 'ZSB'、'IQR' 和 'LOF' 策略 可用的策略包括：
                'ZS'：使用稳健的Z分数，以中位数和中位数绝对偏差（MAD）的函数来检测异常值 
                'IQR'：使用第一四分位数（Q1）和第三四分位数（Q3）加减1.5倍四分位距来检测异常值 
                'LOF'：使用局部异常因子（Local Outlier Factor）来检测异常值

    * verbose: Boolean,  default = 'False' 显示详细信息

    * exclude: str, default = 'None' 排除在离群点检测之外的属性
    """

    def __init__(self, dataset, strategy='ZSB', threshold=0.3,
                 verbose=False, exclude=None):

        self.dataset = dataset

        self.strategy = strategy

        self.threshold = threshold

        self.verbose = verbose

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'threshold': self.threshold,

                'verbose': self.verbose,

                'exclude': self.exclude

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`outlier_detector.get_params().keys()`")

            else:

                setattr(self, k, v)

    def IQR_outlier_detection(self, dataset, threshold):

        X = dataset.select_dtypes(['number'])

        Y = dataset.select_dtypes(exclude=['number'])

        if len(X.columns) < 1:

            print(
                "Error: Need at least one numeric variable for LOF"
                "outlier detection\n Dataset inchanged")

        Q1 = X.quantile(0.25)

        Q3 = X.quantile(0.75)

        IQR = Q3 - Q1

        outliers = X[((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))]

        to_drop = X[outliers.sum(axis=1)/outliers.shape[1] >
                    threshold].index

        to_keep = set(X.index) - set(to_drop)

        if (threshold == -1):

            X = X[~((X < (Q1 - 1.5 * IQR)) |
                  (X > (Q3 + 1.5 * IQR))).any(axis=1)]

        else:

            X = X.loc[list(to_keep)]

        df = X.join(Y)

        print(len(to_drop), "outlying rows have been removed")

        if len(to_drop) > 0:

            if self.verbose:

                print("with indexes:", list(to_drop))

                print()

                print("Outliers:")

                print(dataset.loc[to_drop])

                print()

        return df

    def ZSB_outlier_detection(self, dataset, threshold):
        # Robust Zscore as a function of median and median
        # absolute deviation (MAD)  defined as
        # z-score = |x – median(x)| / mad(x)

        X = dataset.select_dtypes(['number'])

        Y = dataset.select_dtypes(exclude=['number'])

        median = X.apply(np.median, axis=0)

        median_absolute_deviation = 1.4296 * \
            np.abs(X - median).apply(np.median, axis=0)

        modified_z_scores = (X - median) / median_absolute_deviation

        outliers = X[np.abs(modified_z_scores) > 1.6]

        to_drop = outliers[(outliers.count(axis=1) /
                            outliers.shape[1]) > threshold].index

        to_keep = set(X.index) - set(to_drop)

        if (threshold == -1):

            X = X[~(np.abs(modified_z_scores) > 1.6).any(axis=1)]

        else:
            # e.g., remove rows where  40% of variables have zscore
            # above a threshold = 0.4
            X = X.loc[list(to_keep)]

        df = X.join(Y)

        print(len(to_drop), "outlying rows have been removed:")

        if len(to_drop) > 0:

            if self.verbose:

                print("with indexes:", list(to_drop))

                print()

                print("Outliers:")

                print(dataset.loc[to_drop])

                print()

        return df

    def LOF_outlier_detection(self, dataset, threshold):
        # requires no missing value
        # select top 10 outliers
        import pandas as pd
        from sklearn.neighbors import LocalOutlierFactor
        if dataset.isnull().sum().sum() > 0:
            dataset = dataset.dropna()
            print("LOF requires no missing values, so missing values have been removed using DROP.")

        X = dataset.select_dtypes(['number'])

        Y = dataset.select_dtypes(exclude=['number'])
        
        k = int(threshold * 100)

        if len(X.columns) < 1 or len(X) < 1:
            print('Error: Need at least one continuous variable for LOF outlier detection\n Dataset unchanged')
            df = dataset
        else:
            # fit the model for outlier detection (default)
            clf = LocalOutlierFactor(n_neighbors=4, contamination=0.1)
            clf.fit_predict(X)

            # The higher, the more normal.
            LOF_scores = clf.negative_outlier_factor_
            # Inliers tend to have a negative_outlier_factor_ close to -1, while outliers tend to have a larger score.

            if all(score == -1 for score in LOF_scores):
                print("No outliers detected. All data points are considered inliers.")
                df = dataset
            else:
                top_k_idx = np.argsort(LOF_scores)[-k:]
                top_k_values = [LOF_scores[i] for i in top_k_idx]

                # Select data points with LOF scores less than the minimum top_k value
                if top_k_values and top_k_values[0] > -1:  # Check if top_k_values is not empty and the threshold is positive
                    data = X[LOF_scores < top_k_values[0]]
                    to_drop = X[~(LOF_scores < top_k_values[0])].index
                    df = data.join(Y)
                else:
                    print("No outliers detected. The top_k_values is empty or all scores are -1.")
                    df = dataset
                if len(to_drop) > 0:
                    print("{} outlying rows have been removed".format(len(to_drop)))
                    if self.verbose:
                        print("with indexes:", list(to_drop))
                        print()
                        print("Outliers:")
                        print(dataset.loc[to_drop])
                        print()

        return df

    def transform(self):

        start_time = time.time()

        osd = self.dataset

        print()

        print(">>Outlier detection and removal:")
        if not self.dataset.empty:

            d = self.dataset

            if (self.strategy == "ZSB"):

                dn = self.ZSB_outlier_detection(d, self.threshold)

            elif (self.strategy == 'IQR'):

                dn = self.IQR_outlier_detection(d, self.threshold)

            elif (self.strategy == "LOF"):

                dn = self.LOF_outlier_detection(d, self.threshold)

            else:

                        raise ValueError("Threshold invalid. "
                                         "Please choose between "
                                         "'-1' for any outlying value in "
                                         "a row or a value in [0,1] for "
                                         "multivariate outlying row. For "
                                         "example,  with threshold=0.5 "
                                         "if a row has outlying values in "
                                         "half of the attribute set and more, "
                                         "it is considered as an outlier and "
                                         "removed")
            osd = dn


        else:

            print("No outlier detection")

        print("Outlier detection and removal done -- CPU time: %s seconds" %
              (time.time() - start_time))

        print()

        return osd

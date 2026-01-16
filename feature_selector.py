import warnings
import time
import numpy as np
import pandas as pd


class Feature_selector():
    """
   使用特定策略为训练数据集选择特征，并在测试数据集中保持相同的特征。

    Parameters
    ----------
    * dataset: dataframe
    * strategy: 'MR', 'VAR' 和 'LC' 是与任务无关的
                'Tree', 'WR', 'SVC' 用于分类任务
                'L1', 'IMP' 用于回归任务 可用策略：
                'MR': 使用每个变量的缺失比例的默认阈值， 即，移除缺失值比例超过20%（默认值）的变量
                'LC': 检测成对的线性相关变量并移除其中一个
                'VAR': 使用方差阈值
                'Tree': 使用决策树分类作为模型进行特征选择，针对分类任务的目标集
                'SVC': 使用线性SVC作为模型进行特征选择，针对分类任务的目标集
                'WR': 使用 selectKbest (k=10) 和 Chi2 进行特征选择，针对分类任务的目标集
                'L1': 使用 Lasso L1 进行特征选择，针对回归任务的目标集
                'IMP': 使用随机森林回归进行特征选择，针对回归任务的目标集

    * exclude: str, default = 'None' 要从特征选择中排除的变量名称。

    * threshold: float, default = '0.3' 适用于 MR, VAR, LC, L1, and IMP

    * verbose: Boolean,  default = 'False' 是否显示处理信息
    """

    def __init__(self, dataset, target , strategy='LC', exclude=None,
                 threshold=0.3, verbose=False):

        self.dataset = dataset
        
        self.target = target

        self.strategy = strategy

        self.exclude = exclude

        self.threshold = threshold

        self.verbose = verbose

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'exclude': self.exclude,

                'threshold':  self.threshold,

                'verbose': self.verbose

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`feature_selector.get_params().keys()`")

            else:

                setattr(self, k, v)

    # Feature selection based on missing values
    def FS_MR_missing_ratio(self, dataset, missing_threshold=.2):

        print("Apply MR feature selection with missing "
              "threshold=", missing_threshold)

        # Find the features with a fraction of missing
        # values above `missing_threshold`
        # Calculate the fraction of missing in each column
        missing_series = dataset.isnull().sum() / dataset.shape[0]

        missing_stats = pd.DataFrame(missing_series).rename(
            columns={'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        missing_stats = missing_stats.sort_values(
            'missing_fraction', ascending=False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series >
                                      missing_threshold]).reset_index().\
            rename(columns={'index': 'feature', 0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        if self.verbose:

            print(missing_stats)

        print('%d features with greater than %0.2f missing values.\n' %
              (len(to_drop), missing_threshold))

        print('List of variables to be removed :', to_drop)

        to_keep = set(dataset.columns) - set(to_drop)

        if self.verbose:

            print("List of variables to be keep")

            print(list(to_keep))

        return dataset[list(to_keep)]

    def FS_LC_identify_collinear(self, dataset, correlation_threshold=0.8):

        # Finds linear-based correlation between features (LC)
        # For each pair of features with a correlation
        # coefficient greather than `correlation_threshold`,
        # only one of the pair is identified for removal.
        # one attribute can be kept (excluded from feature selection)
        # Using code adapted from: https://gist.github.com/
        # Swarchal/e29a3a1113403710b6850590641f046c
        print("Apply LC feature selection with threshold=",
              correlation_threshold)

        # Calculate the correlations between every column
        corr_matrix = dataset.corr()

        if self.verbose:

            print("Correlation matrix")

            print(corr_matrix)

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(
            upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(
            columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop
        for column in to_drop:

            # Find the correlated features
            corr_features = list(
                upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(
                upper[column][upper[column].abs() > correlation_threshold])

            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = pd.concat([record_collinear, temp_df], ignore_index=True)

        print('%d features with linear correlation greater than %0.2f.\n' %
              (len(to_drop), correlation_threshold))


        print('List of correlated variables to be removed :', to_drop)

        to_keep = set(dataset.columns) - set(to_drop)

        if self.verbose:

            print("List of numerical variables to be keep")

            print(list(to_keep))

        return dataset[list(to_keep)]

    def FS_WR_identify_best_subset(self, df_train, df_target, k=10):
        # k number of k best features to keep
        # Feature extraction
        # Requires non-missing values
        # Requires a categorical target variable
        from sklearn.feature_selection import SelectKBest, chi2
        print("Apply WR feature selection")

        if df_train.isnull().sum().sum() > 0:
            df_train = df_train.dropna()
            print('WR requires no missing values, so missing values have been removed applying DROP on the train dataset.')

        X = df_train.select_dtypes(include=[np.number])

        if len(X.columns) < 1:
            if self.verbose:
                print('Error: Need at least one continuous variable for identifying the best subset of features')
            df = df_train
        elif len(X.columns) <= k:
            if self.verbose:
                print(f'Number of features ({len(X.columns)}) <= k ({k}), returning all features')
            return X
        else:
            # 调整k值以确保不超过特征数量
            k = min(k, len(X.columns))
            # Filter out negative variables
            negative_vars = X.columns[(X < 0).any()].tolist()
            if len(negative_vars) == len(X.columns):
                if self.verbose:
                    print("Input dataset has no positive variables. WR feature selection is not applicable. Dataset unchanged.")
                df = df_train
            else:
                if len(negative_vars) > 0:
                    if self.verbose:
                        print(f"Negative variables detected: {negative_vars}. WR feature selection is only applied to positive variables.")
                    X = X[X.columns.difference(negative_vars)]

                if len(X.columns) == 0:
                    if self.verbose:
                        print("After removing negative variables, no features left for WR feature selection.")
                    df = df_train
                else:
                    # 再次检查并调整k值，因为可能移除了负值特征
                    k = min(k, len(X.columns))
                    selector = SelectKBest(score_func=chi2, k=k)
                    selector.fit(X, df_target)

                    Best_Flist = X.columns[selector.get_support(indices=True)].tolist()

                    if self.verbose:
                        print("Best features to keep:", Best_Flist)

                    df = X[Best_Flist]

        return df

    def FS_SVC_based(self, df_train , df_target):
        # Feature extraction
        # requires non missing value

        from sklearn.svm import LinearSVC

        from sklearn.feature_selection import SelectFromModel

        print("Apply SVC feature selection")
        
        if df_train.isnull().sum().sum() > 0:

            df_train = df_train.dropna()

            print('SVC requires no missing values, '
                  'so missing values have been removed applying '
                  'DROP on the train dataset.')
        df = df_train
        if len(df_train.columns) < 1 or len(df_train) < 1:

            print(
                  'Error: Need at least one continous variable '
                  'for feature selection \n Dataset inchanged')

            df = df_train

        else:

            X = df_train.select_dtypes(['number'])

            Y = df_target

            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)

            model = SelectFromModel(lsvc, prefit=True)

            Best_Flist = X.columns[model.get_support(indices=True)].tolist()

            df = df[Best_Flist]

            if self.verbose:

                print("Best features to keep", Best_Flist)

        return df

    def FS_Tree_based(self, df_train , df_target):
        # Feature extraction using the decision tree classification as model
        # requires non missing value

        from sklearn.ensemble import ExtraTreesClassifier

        from sklearn.feature_selection import SelectFromModel

        print("Apply Tree-based feature selection ")
        

        if df_train.isnull().sum().sum() > 0:

            df_train = df_train.dropna()

            print('Tree requires no missing values, so missing '
                  'values have been removed applying DROP '
                  'on the train dataset.')
        df = df_train
        if len(df_train.columns) < 1 or len(df_train) < 1:

            print('Error: Need at least one continous variable '
                  'for feature selection \n Dataset inchanged')

            df = df_train

        else:

            X = df_train.select_dtypes(['number'])

            Y = df_target

            clf = ExtraTreesClassifier(n_estimators=50)

            clf = clf.fit(X, Y)

            model = SelectFromModel(clf, prefit=True)

            Best_Flist = X.columns[model.get_support(indices=True)].tolist()

            if self.verbose:

                print("Best features to keep", Best_Flist)

            df = df[Best_Flist]

        return df

    def transform(self):
        df = self.dataset.copy()
        df1 = df.drop(self.target,axis=1)
        start_time = time.time()

        to_keep = []

        print()

        print(">>Feature selection ")

        print("Before feature selection:")
        print(df1.shape[1], "features ")

        if (self.strategy == "MR"):
            dn = self.FS_MR_missing_ratio(df1, missing_threshold=self.threshold)
        elif (self.strategy == "LC"):
            d = df1.select_dtypes(['number'])
            do = df1.select_dtypes(exclude=['number'])
            dn = self.FS_LC_identify_collinear(d, correlation_threshold=self.threshold)
            dn = pd.concat([dn, do], axis=1)
        elif (self.strategy == 'VAR'):
            dn = df1.select_dtypes(['number'])
            coef = dn.std()
            print("Apply VAR feature selection with threshold=", self.threshold)
            abstract_threshold = np.percentile(coef, 100. * self.threshold)
            to_discard = coef[coef < abstract_threshold].index.tolist()
            dn.drop(to_discard, axis=1, inplace=True)
        else:
            dn = df1.select_dtypes(['number'])
            if self.target is not None:
                dt = self.dataset[self.target]
                if dn.isnull().sum().sum() > 0:
                    dn = dn.dropna()
                    dt = dt.loc[dn.index]
                    print('Warning: This strategy requires no missing values, so missing values have been removed applying DROP on the dataset.')
                else:
                    dt = self.dataset[self.target].loc[dn.index]

                if (self.strategy == 'L1'):
                    from sklearn.linear_model import Lasso
                    print("Apply L1 feature selection with threshold=", self.threshold)
                    model = Lasso(alpha=100.0, tol=0.01, random_state=0)
                    model.fit(dn, dt)
                    coef = np.abs(model.coef_)
                    abstract_threshold = np.percentile(coef, 100. * self.threshold)
                    to_discard = dn.columns[coef < abstract_threshold].tolist()
                    dn.drop(to_discard, axis=1, inplace=True)
                elif (self.strategy == 'IMP'):
                    from sklearn.ensemble import RandomForestRegressor
                    print("Apply IMP feature selection with threshold=", self.threshold)
                    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=0)
                    model.fit(dn, dt)
                    coef = model.feature_importances_
                    abstract_threshold = np.percentile(coef, 100. * self.threshold)
                    to_discard = dn.columns[coef < abstract_threshold].tolist()
                    dn.drop(to_discard, axis=1, inplace=True)
                elif (self.strategy == "Tree"):
                    dn = self.FS_Tree_based(dn, dt)
                elif (self.strategy == "WR"):
                    dn = self.FS_WR_identify_best_subset(dn, dt)
                elif (self.strategy == "SVC"):
                    dn = self.FS_SVC_based(dn, dt)
                else:
                    print("Strategy invalid. Please choose between 'Tree', 'WR', 'SVC', 'VAR', 'LC' or 'MR' -- No feature selection done on the dataset")
                    dn = df.copy()
            else:
                print("No target provided. Please choose a strategy among 'VAR', 'LC', 'MR'")
                dn = df.copy()

        to_keep = [column for column in dn.columns]
        to_keep.append(self.target)
        df=df[to_keep]
        print(df.columns)

        print("After feature selection:")
        print(len(to_keep), "features remain")
        print(to_keep)
        print("Feature selection done -- CPU time: %s seconds" % (time.time() - start_time))
        print()

        return df
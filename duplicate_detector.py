import warnings
import numpy as np
import jellyfish as jf
import pandas as pd
pd.options.mode.chained_assignment = None


def add_key_reindex(dataset, rand=False):
    if rand:
        dataset = dataset.reindex(np.random.permutation(dataset.index))
    dataset['New_ID'] = range(1, 1+len(dataset))
    dataset['New_ID'].apply(str)
    return(dataset)
class Duplicate_detector():
    '''
    从数据集中移除重复记录
    参数
    ----------
    * dataset: 输入的数据框
    * threshold: 浮点数，默认值为'0.6'，仅用于'AD'策略
    *  strategy: 字符串，默认为'ED'
       去重策略的选择：'ED', 'MARK','AP'
       'ED': 精确重复记录检测/移除
       'MARK':  标记重复的行 
       'AP': 使用在'strategy'中指定的特定距离：
               'DL'（默认）用于Damerau Levenshtein距离
               'LM' 用于Levenshtein距离
               'JW' 用于Jaro-Winkler距离
    * metric: 字符串，默认为'DL' 仅用于'AD'策略
    * verbose: 布尔值，默认为'False'，否则显示已移除的重复行列表
    * exclude: 字符串，默认为'None' 要从去重中排除的变量名称
    '''
    def __init__(self, dataset, strategy='ED', threshold=0.6,
                   verbose=False, exclude=None):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold
        self.verbose = verbose
        self.exclude = exclude  
    def get_params(self, deep=True):
        return {'strategy': self.strategy,
                'threshold':  self.threshold,
                'verbose': self.verbose,
                'exclude': self.exclude
                }
    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`duplicate_detector.get_params().keys()`")
            else:
                setattr(self, k, v)
    def ED_Exact_duplicate_removal(self, dataset):
        #精准匹配去重
        if not dataset.empty:
            df = dataset.drop_duplicates()
            print('Initial number of rows:', len(dataset))
            print('After deduplication: Number of rows:', len(df))
        else:
            print("No duplicate detection, empty dataframe")
        return df
    def AD_Approx_string_duplicate_removal(self, dataset,
                                           threshold, strategy):
        '''
        近似匹配去重
        Parameters
        ----------
        dataset : dataframe
            输入的数据.
        threshold : float
            近似匹配的程度.
        metric : str ,  'DL'（默认）用于Damerau Levenshtein距离
                        'LM' 用于Levenshtein距离
                        'JW' 用于Jaro-Winkler距离
            DESCRIPTION. The default is "DL".
        Returns
        -------
        df : dataframe
            处理后的数据.
        '''
        dataset = add_key_reindex(dataset, rand=True)
        data = dataset.applymap(str)
        data = data.apply(lambda x: '*'.join(x.values.tolist()),
                          axis=1)
        data = data.astype(str)
        data = data.str.replace(" ", "")
        # delchars = ''.join(c for c in map(chr,
        # range(256)) if not c.isalnum())
        for row in data.index:
            data[row] = data[row].lower()
        out = pd.DataFrame(columns=["Dup_ID1", "Dup_ID2", "Dup_1", "Dup_2"])
        if strategy == "DL":  # Damerau Levenshtein Distance
            res = {_d: [] for _d in data}
            for _d in res.keys():
                for row in data.index:
                    if _d != data[row] \
                        and jf.damerau_levenshtein_distance(_d, data[row]) < \
                            ((len(_d) + len(data[row])/2)*threshold):
                        res[_d].append(data[row])
                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])
        elif strategy == "LM":  # Levenshtein Distance
            res = {_d: [] for _d in data}
            for _d in res.keys():
                for row in data.index:
                    if _d != data[row] \
                        and jf.levenshtein_distance(_d, data[row]) < \
                            ((len(_d) + len(data[row])/2)*threshold):
                        res[_d].append(data[row])
                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])
        elif strategy == "JW":  # Jaro-Winkler Distance
            res = {_d: [] for _d in data}
            for _d in res.keys():
                for row in data.index:
<<<<<<< HEAD
                    if _d != data[row] and jf.jaro_winkler(_d, data[row]) >  \
=======
                    if _d != data[row] and jf.jaro_winkler_similarity(_d, data[row]) >  \
>>>>>>> 2d759d3 (更新AutoGA项目代码)
                            ((len(_d) + len(data[row])/2)*threshold):
                        res[_d].append(data[row])
                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])
        filtered = {k: v for k, v in res.items() if len(v) > 0}
        out = out[~out[["Dup_ID1", "Dup_ID2"]].apply(
            frozenset, axis=1).duplicated()]
        out.reset_index(drop=True, inplace=True)
        # d = dataset['New_ID'].astype(str)
        if self.verbose:
            print("Duplicates IDs:", out)
            dups = pd.DataFrame.from_dict(filtered, orient='index')
            print("Duplicates:", dups)
            print("Duplicates removed: ",
                  dataset[dataset['New_ID'].isin(out['Dup_ID2'])])
        df = dataset[~dataset['New_ID'].isin(out['Dup_ID2'])]
        print("Number of duplicate rows removed:", len(dataset)-len(df))
        return df

    def MARK_Duplicate_marking(self,dataset):
         marked_df = dataset
         marked_df['is_duplicate'] = marked_df.duplicated(keep=False)
         if self.verbose:
             print(f'Number of duplicates marked: {marked_df["is_duplicate"].sum()}')
         return marked_df
    
    def transform(self):
        dn = self.dataset
        print(">>Duplicate detection and removal:")
        if not self.dataset.empty:
            if (self.strategy == "ED"):
                    dn = self.ED_Exact_duplicate_removal(self.dataset)
            elif (self.strategy == "DL"):
                    dn = self.AD_Approx_string_duplicate_removal(
                                self.dataset,
                                strategy=self.strategy,
                                threshold=self.threshold)
            elif (self.strategy == "LM"):
                    dn = self.AD_Approx_string_duplicate_removal(
                                self.dataset,
                                strategy=self.strategy,
                                threshold=self.threshold)
            elif (self.strategy == "JW"):
                    dn = self.AD_Approx_string_duplicate_removal(
                                self.dataset,
                                strategy=self.strategy,
                                threshold=self.threshold)
            elif self.strategy == 'MARK':
                    dn = self.MARK_Duplicate_marking(self.dataset)

            else:
                raise ValueError("Strategy invalid."
                                 "Please choose between "
                                 "'ED', 'METRIC' or 'AD'")
        return dn

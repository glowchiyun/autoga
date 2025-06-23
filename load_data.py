import numpy as np
import pandas as pd
<<<<<<< HEAD
=======
import os
from sklearn.datasets import fetch_openml
import logging
>>>>>>> 2d759d3 (更新AutoGA项目代码)

def load_data(file_path):
    '''
    读取数据
    Parameters
    ----------
    file_path : str
<<<<<<< HEAD
        数据存储路径.
=======
        数据存储路径或OpenML数据集ID.
>>>>>>> 2d759d3 (更新AutoGA项目代码)
    Returns
    -------
    data : dataframe
        返回读取后的文件
    '''
<<<<<<< HEAD
    path=str(file_path)
    try:
        if '.csv' in path:
            data = pd.read_csv(path,encoding='latin-1')
            print("Lode file success.")
        elif '.xls' in path:
            data = pd.read_excel(path,encoding='latin-1')
            print("Lode file success.")
        elif '.h5' in path:
            data = pd.read_hdf(path)
        else:
            print("Unsupported file type.")
            return None
        return data
    except Exception as e:
        print("Error loading data: ", e)
        return None
    #删除未命名的列
    try:
        del data["Unnamed: 0"]
    except Exception:
        pass
    
=======
    path = str(file_path)
    try:
        # 检查是否是OpenML数据集ID
        if path.isdigit():
            logging.info(f"从OpenML下载数据集ID: {path}")
            try:
                data = fetch_openml(data_id=int(path), as_frame=True)
                logging.info(f"成功下载数据集: {data.details['name']}")
                return data.frame
            except Exception as e:
                logging.error(f"从OpenML下载数据失败: {str(e)}")
                return None
        
        # 检查文件是否存在
        if not os.path.exists(path):
            logging.error(f"文件不存在: {path}")
            return None
            
        # 根据文件类型读取数据
        if '.csv' in path:
            data = pd.read_csv(path, encoding='latin-1')
            logging.info("成功读取CSV文件")
        elif '.xls' in path:
            data = pd.read_excel(path, encoding='latin-1')
            logging.info("成功读取Excel文件")
        elif '.h5' in path:
            data = pd.read_hdf(path)
            logging.info("成功读取HDF5文件")
        else:
            logging.error("不支持的文件类型")
            return None
            
        # 删除未命名的列
        try:
            del data["Unnamed: 0"]
        except Exception:
            pass
            
        return data
        
    except Exception as e:
        logging.error(f"加载数据时发生错误: {str(e)}")
        return None

>>>>>>> 2d759d3 (更新AutoGA项目代码)
def auto_detect_dtypes(df):
    '''
    自动探查并转换数据类型
    Parameters
    ----------
    df : dataframe类型
        转换前的dataframe
    Returns
    -------
    df : dataframe类型
        转换后的dataframe
    '''
    type_changes = {}  # 创建一个字典来记录类型变化
    for col in df.columns:
        # 类别类型推断，如果唯一值的数量小于行数的10%
        unique_values_count = len(df[col].dropna().unique())
        original_dtype = df[col].dtype  # 保存原始数据类型
        # # 如果是整数类型，并且只有两个唯一值，1和0，则转换为布尔类型
        # if df[col].dtype == 'int64' and df[col].nunique() == 2 and set(df[col]).issubset({1, 0}):
        #     df[col] = df[col].astype(bool)
        
        # # 检查object类型列
        # elif df[col].dtype == 'object':
        #     # 布尔类型推断，考虑整数和小写/大写字符串
        #     if df[col].nunique() == 2 and set(df[col]).issubset({True, False, 1, 0}):
        #         df[col] = df[col].astype(bool)
            # 日期时间类型推断
        if any(char in set(df[col]) for char in ['-', '/']):
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                pass
            # 时间类型推断
        elif all(char in set(df[col]) for char in [':', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            try:
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S').dt.time
            except ValueError:
                pass
            # 类别类型推断
        elif unique_values_count <= len(df) * 0.1 and df[col].dtype == 'object' or  df[col].dtype =='category':
                df[col+'coded'] = pd.factorize(df[col])[0]
        # 如果数据类型发生了变化，记录下来
        new_dtype = df[col].dtype
        if original_dtype != new_dtype:
            type_changes[col] = f"{original_dtype}--->{new_dtype}"

    # # 将剩余的object类型转换为string类型
    # for col in df.select_dtypes(include=['object']).columns:
    #     df[col] = df[col].astype('string')
    #     # 如果数据类型发生了变化，记录下来
    #     if df[col].dtype != original_dtype:
    #         type_changes[col] = f"{original_dtype}--->{new_dtype}"
    print("类型转换的属性：")
    # 打印类型变化
    for col, change in type_changes.items():
        print(f"{col}: {change}")
    return df

def profile_summary(dataset):
    '''
    生成数据摘要信息
    Parameters
    ----------
    dataset : dataframe类型
        输入的dataframe
    Returns
    -------
    pf : dataframe类型
        输出的数据概况信息
    '''
    # 初始化一个空的DataFrame，用于存储概况摘要
    pf = pd.DataFrame(columns=[ 'Type', 'Missing', 'Unique',
                               'Mean', 'Variance', 'Std Dev', 
                                'Max', 'Min'])
    
    # 遍历数值型列
    for attribute in dataset.select_dtypes(include=[np.number]).columns:
        att_type = dataset[attribute].dtype
        num_missing = dataset[attribute].isnull().sum()
        unique_values = dataset[attribute].nunique()
        mean_value = dataset[attribute].mean()
        variance_value = dataset[attribute].var()
        std_dev_value = dataset[attribute].std()
        max_value = dataset[attribute].max()
        min_value = dataset[attribute].min()
        # 将结果作为新行添加到概况摘要DataFrame中
        pf.loc[attribute] = {
            'Type': att_type,
            'Missing': num_missing,
            'Unique': unique_values,
            'Mean': mean_value,
            'Variance': variance_value,
            'Std Dev': std_dev_value,
            'Max': max_value,
            'Min': min_value,
        }
    
    # 遍历非数值型列，不计算数值统计属性
    for attribute in dataset.select_dtypes(exclude=[np.number]).columns:
        att_type = dataset[attribute].dtype
        num_missing = dataset[attribute].isnull().sum()
        unique_values = dataset[attribute].nunique()
        # 非数值型列的数值统计属性为N/A
        pf.loc[attribute] = {
            'Type': att_type,
            'Missing': num_missing,
            'Unique': unique_values,
            'Mean': 'N/A',
            'Variance': 'N/A',
            'Std Dev': 'N/A',
            'Max': 'N/A',
            'Min': 'N/A',
        }
    
    return pf

#! python -m GAFactor.treatment.py

import pandas as pd
import numpy as np
import copy
from sklearn.impute import SimpleImputer
from sklearn import preprocessing





def missing_correct(data:pd.DataFrame, method:str,fill_value=None) -> pd.DataFrame:
    """
    对缺失值进行处理
    ：data: 预处理的数据,pd.DataFrame:时间与股票代码双索引数据
    ：method： 处理方式，可以选择'median'->     中值填充
                             'mean'->       均值填充
                             ‘most_frequent’-> 众数填充
                             ‘constant’—> 将空值填充为自定义的值，需要配合fill_value使用，fill_value即为替换值,默认为0
    """
    imputer = SimpleImputer(strategy =  method,fill_value=fill_value)
    imputer.fit(data)
    data_trans = imputer.transform(data)
    data_fill = pd.DataFrame(data_trans,index=data.index,columns=data.columns)
    return data_fill


def median_correct(data:pd.DataFrame, n=5)->pd.DataFrame:
    """
    中位数去极值
    绝对中位差（MAD）较标准差而言对离群值（outlier）也就是异常值更加稳健，
    标准差计算中，数据点到其均值的距离要求平方，
    因此对偏离较为严重的点偏离的影响得以加重，
    也就是说异常值严重影响着标准差的求解，
    因此用中位数极值法能更好的度量和修正异常值。
    :param data: numpy.array, 需要去极值的原始数据
    :param n: int, 中位数的倍数
    :return: numpy.array, 去极值修正后的数据
    """
    md = np.median(data[~np.isnan(data)])
    MAD = np.median(np.abs(data[~np.isnan(data)] - md))
    up = md + n * MAD
    down = md - n * MAD
    new_data = copy.deepcopy(data)
    new_data[new_data > up] = up
    new_data[new_data < down] = down
    return new_data


def standard_correct(data:pd.DataFrame)-> pd.DataFrame:
    """
    对数据进行标准化
    data: 预处理的数据,pd.DataFrame:时间与股票代码双索引数据
    """
    data_scaled = preprocessing.scale(data)
    data_standard = pd.DataFrame(data_scaled,index = data.index,columns=data.columns)
    return data_standard


def min_max_correct(data:pd.DataFrame,feature_range=None)-> pd.DataFrame:
    """
    对数据进行归一化
    data: 预处理的数据,pd.DataFrame:时间与股票代码双索引数据
    feature_range: 可将数据缩小为特性范围，输入为列表，如[0,2]代表数据缩小到0至2之间
    """

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
    data_scaled = min_max_scaler.fit_transform(data)
    data_normalized = pd.DataFrame(data_scaled,index = data.index,columns=data.columns)
    return data_normalized

def normalize_correct(data, opt)-> pd.DataFrame:
    """
    对数据进行正则化
    将每个样本缩放到单位范数(每个样本的范数为1)
    主要思想：对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数(l1-norm,l2-norm)等于1
    可以选择
    data: 预处理的数据,pd.DataFrame:时间与股票代码双索引数据
    opt:  正则化可以选择的三种方式，分别为'l1','l2', 'max'
    """
    opt1 = opt
    if opt1 == 'l1':
        data_normalized = preprocessing.normalize(data, norm='l1')
    if opt1 == 'l2':
        data_normalized = preprocessing.normalize(data, norm='l2')
    if opt1 == 'max':
        data_normalized = preprocessing.normalize(data, norm='max')
    data_final = pd.DataFrame(data_normalized,index=data.index, columns=data.columns)
    return data_final






if __name__ == '__main__':
    data = pd.read_excel(r'/Users/taiyunshuai/Desktop/stock_data.xlsx',index_col = [0,1])
    #print(missing_correct(data,'constant',1))
    #print(median_correct(data))
    #print(standard_correct(data))
    #print(min_max_correct(data,[0,10]))
    print(normalize_correct(missing_correct(data,'constant',1),'max'))
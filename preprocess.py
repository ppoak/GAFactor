# %%
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


# def standard_correct(data:pd.DataFrame)-> pd.DataFrame:
    # """
    # 对数据进行标准化
    # data: 预处理的数据,pd.DataFrame:时间与股票代码双索引数据
    # """
    # data_scaled = preprocessing.scale(data)
    # data_standard = pd.DataFrame(data_scaled,index = data.index,columns=data.columns)
    # return data_standard

def zscore_standard(data):
    """
    ZScore标准化函数
    原始值减均值后除以标准差，使分布尽可能逼近N(0, 1)分布
    :param data: numpy.array, 待标准化的原值
    :return: numpy.array, 标准化后的数据
    """
    mean = np.mean(data[~np.isnan(data)])
    std = np.std(data[~np.isnan(data)])
    new_data = (data - mean) / std
    return new_data

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


def split_data(data:pd.DataFrame, train_start:str, train_end:str, test_start:str, test_end:str)->pd.DataFrame:
    """
    对数据按照设定时间进行

    train_start:：训练集开始时间例：‘2021-01-01’
    train_end:训练集结束时间
    test_start:测试集开始时间
    test_end:测试集结束时间
    """
    train = data.copy()[(data.index >= train_start) & (data.index < train_end)]
    test = data.copy()[(data.index >= test_start) & (data.index < test_end)]
    return train,test


def data_preprocess(data:pd.DataFrame,train_start:str,train_end:str,test_start:str,test_end:str,standardize=False):
    """
    实现功能：根据设置的train_span,test_span，以及是否标准化对数据进行处理
    train_start:：训练集开始时间例：‘2021-01-01’
    train_end:训练集结束时间
    test_start:测试集开始时间
    test_end:测试集结束时间
    standardize:是否标准化参数
    """
    train,test = split_data(data,train_start,train_end,test_start,test_end)
    if standardize == True:
        train_1  = train.groupby('date').apply(zscore_standard).squeeze().unstack()
        test_1   = test.groupby('date').apply(zscore_standard).squeeze().unstack()
    else:
        train_1 = train.squeeze().unstack()
        test_1 = test.groupby('date').apply(zscore_standard).squeeze().unstack()
    return train_1,test_1



if __name__ == '__main__':
    #data = pd.read_excel(r'stock_data.xlsx',index_col = [0,1])
    #print(missing_correct(data,'constant',1))
    #print(median_correct(data))
    #print(standard_correct(data))
    #print(min_max_correct(data,[0,10]))
    #print(normalize_correct(missing_correct(data,'constant',1),'max'))
    data = pd.read_parquet(r'data/kline_daily.parquet')
    grp_col = 'stock_code'
    #开盘价
    # adj_open_1 = pd.DataFrame(data.loc[:, "open"] * data.loc[:, "back_adjfactor"],columns=['adj_open'])
    # adj_open_2 = adj_open_1.groupby(grp_col).apply(missing_correct,'constant',0)
    # adj_open = adj_open_2.groupby(grp_col).apply(standard_correct)
    # adj_open.squeeze().unstack().to_parquet('data/raw_factor/adj_open.parquet')
    # #收盘价
    # adj_close_1 = pd.DataFrame(data.loc[:, "close"] * data.loc[:, "back_adjfactor"],columns=['adj_close'])
    # adj_close_2 = adj_close_1.groupby(grp_col).apply(missing_correct,'constant',0)
    # adj_close = adj_close_2.groupby(grp_col).apply(standard_correct)
    # adj_close.squeeze().unstack().to_parquet('data/raw_factor/adj_close.parquet')
    #最高价
    # adj_high_1 = pd.DataFrame(data.loc[:, "high"] * data.loc[:, "back_adjfactor"],columns=['adj_high'])
    # adj_high_2 = adj_high_1.groupby(grp_col).apply(missing_correct,'constant',0)
    # adj_high = adj_high_2.groupby(grp_col).apply(standard_correct)
    # adj_high.squeeze().unstack().to_parquet('data/raw_factor/adj_high.parquet')
    # #最低价
    # adj_low_1 = pd.DataFrame(data.loc[:, "low"] * data.loc[:, "back_adjfactor"],columns=['adj_low'])
    # adj_low_2 = adj_low_1.groupby(grp_col).apply(missing_correct,'constant',0)
    # adj_low = adj_low_2.groupby(grp_col).apply(standard_correct)
    # adj_low.squeeze().unstack().to_parquet('data/raw_factor/adj_low.parquet')
    # #volume
    volume_1 = pd.DataFrame(data.loc[:, "volume"],columns=['volume'])
    volume_2 = volume_1.groupby(grp_col).apply(missing_correct,'constant',0)
    volume = volume_2.groupby(grp_col).apply(standard_correct)
    volume.squeeze().unstack().to_parquet('data/raw_factor/volume.parquet')
    #label
    label_1 = pd.DataFrame((data.groupby(level=1)['open'].shift(-5) / data.groupby(level=1)['open'].shift(-1) - 1).rename('label')) 
    label_2 = label_1.groupby(grp_col).apply(missing_correct,'constant',0)
    label = label_2.groupby(grp_col).apply(standard_correct)
    label.squeeze().unstack().to_parquet('data/raw_factor/label.parquet')


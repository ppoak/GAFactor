import warnings
import pandas as pd
import lightgbm as lgb
from operators import *


def compute_feature(exprs, dataset: str = 'train'):
    exprs = list(map(lambda x: x.replace('open', f'{dataset}_open')\
        .replace('high', f'{dataset}_high')\
        .replace('low', f'{dataset}_low')\
        .replace('close', f'{dataset}_close')\
        .replace('vol', f'{dataset}_volume'), exprs))
    features = list(map(lambda x: pd.DataFrame(eval(x), 
        index=eval(dataset).index.levels[0], 
        columns=eval(dataset).index.levels[1]).stack(), exprs
    ))
    features = pd.concat(features, axis=1).dropna()
    features = features.add_prefix('feature_')
    return features

warnings.filterwarnings('ignore')

train = pd.read_parquet('data/train_dataset.parquet')
test = pd.read_parquet('data/test_dataset.parquet')
train_open, train_high, train_low, train_close, train_volume, train_label = (
    train['open'].unstack().values,
    train['high'].unstack().values,
    train['low'].unstack().values,
    train['close'].unstack().values,
    train['volume'].unstack().values,
    train['label'],
)
test_open, test_high, test_low, test_close, test_volume, test_label = (
    test['open'].unstack().values,
    test['high'].unstack().values,
    test['low'].unstack().values,
    test['close'].unstack().values,
    test['volume'].unstack().values,
    test['label'],
)

with open('result/efficient.txt', 'r') as f:
    exprs = f.readlines()
exprs = list(map(lambda x: x.split(';')[0], exprs))
train_features = compute_feature(exprs, 'train')
test_features = compute_feature(exprs, 'test')
regressor = lgb.LGBMRegressor()
regressor.fit(train_features, train_label.loc[train_features.index])
train_pred = regressor.predict(train_features)
test_pred = regressor.predict(test_features)
train_ic = pd.Series(train_pred, index=train_features.index).groupby(level=0).corr(train_label.loc[train_features.index]).mean()
test_ic = pd.Series(test_pred, index=test_features.index).groupby(level=0).corr(test_label.loc[test_features.index]).mean()
print(f'Train IC: {train_ic}\tTest IC: {test_ic}')

import numba
import numpy as np


def raw(a):
    return a

def abs(a: np.ndarray):
    return np.abs(a)

def sign(a: np.ndarray):
    return np.sign(a)

def log(a: np.ndarray):
    return np.log(a)

def sqrt(a: np.ndarray):
    if isinstance(a, (int, float)):
        if a < 0:
            a = -a
    else:
        a = np.abs(a)
    return np.sqrt(a)

def ssqrt(a: np.ndarray):
    if isinstance(a, (int, float)):
        if a < 0:
            a = -a
            sign = -1
        else:
            sign = 1
    else:
        sign = np.ones_like(a)
        sign[a<0] = -1
        a = np.abs(a)
    return sign * np.sqrt(a)

def square(a: np.ndarray):
    return a ** 2

def add(a: np.ndarray, b: 'int | float'):
    return a + b

def sub(a: np.ndarray, b: 'int | float'):
    return a -b

def mul(a: np.ndarray, b: 'int | float'):
    return a * b

def div(a: np.ndarray, b: 'int | float'):
    if isinstance(b, (int, float)):
        if b >= 0:
            b = max(1e-2, b)
        else:
            b = min(-1e-2, b)
    return a / b

def power(a: np.array, b: float):
    return np.power(a, b)

def sum(a: np.ndarray, d: int):
    mat = np.zeros((a.shape[0], a.shape[0]-d+1))
    for i in range(mat.shape[0]-d+1):
        mat[i:(i + d), i] = 1
    res =  a.T @ mat
    res = np.hstack((np.full((a.shape[1], d-1), fill_value=np.nan), res))
    return res.T

def mean(x: np.ndarray, d: int) -> np.ndarray:
    ma = np.zeros((x.shape[0], x.shape[0]-d+1))
    for i in range(ma.shape[0]-d+1):
        ma[i:(i + d), i] = 1/d
    res =  x.T @ ma
    res = np.hstack((np.full((x.shape[1], d-1), fill_value=np.nan), res))
    return res.T

def var(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        v = x[row:row + d].var(axis=0)
        res[row + d - 1, :] = v
    return res

def skew(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        tmp = x[row:row + d]
        s = ((tmp - tmp.mean(axis=0)) ** 3).mean(axis=0)
        res[row + d - 1, :] = s
    return res

def kurt(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        tmp = x[row:row + d]
        k = ((tmp - tmp.mean(axis=0)) ** 4).mean(axis=0) / (tmp.var(axis=0) ** 2)
        res[row + d - 1, :] = k
    return res

def max_(x: np.array, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        m = x[row:row + d].max(axis=0)
        res[row + d - 1, :] = m
    return res

def min_(x: np.array, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        m = x[row:row + d].min(axis=0)
        res[row + d - 1, :] = m
    return res

def delta(x: np.array, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        tmp = x[row:row + d]
        res[row + d - 1, :] = tmp[-1, :] - tmp[0, :]
    return res

def delay(x: np.ndarray, d: int) -> np.ndarray:
    res = np.roll(x, d, axis=0)
    res[:d, :] = np.nan
    return res

def rank(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        rank = x[row:row + d].argsort(axis=0).argsort(axis=0)
        res[row + d - 1, :] = rank[-1, :]
    return res

def std(x: np.ndarray, d: int):
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        res[row + d - 1, :] = np.std(x[row:row + d], axis=0)
    return res

def decay_linear(x: np.ndarray, d:int):
    raise NotImplementedError("Not Implemented")
    ema = np.zeros([x.shape[0], x.shape[0]-d+1])

def correlation(x: np.ndarray, y: np.ndarray, d: int):
    raise NotImplementedError("Not Implemented")
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        x_demean = np.nan_to_num(x - np.nanmean(x[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        y_demean = np.nan_to_num(y - np.nanmean(y[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        res[row + d - 1, :] =(x_demean.T @ y_demean).diagonal() \
                / (np.linalg.norm(x_demean, axis=0) * np.linalg.norm(y_demean, axis=0))
    return res

def covariance(x: np.ndarray,y: np.ndarray,d: int):
    raise NotImplementedError("Not Implemented")
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        x_demean = np.nan_to_num(x - np.nanmean(x[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        y_demean = np.nan_to_num(y - np.nanmean(y[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        res[row + d - 1, :] = (x_demean.T @ y_demean).diagonal() / (d - 1)
    return res

if __name__ == '__main__':
    np.random.seed(0)
    d = np.array([
        [1, 2, 4, 2, 3],
        [3, 2, 2, 1, 5],
        [2, 4, 3, 1, 5],
        [2, 3, 3, 1, 4],
    ], dtype='float32')
    print(std(d, 2))


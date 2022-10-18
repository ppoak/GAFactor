import numba
import numpy as np


def add(a, b):
    return a + b

def sub(a, b):
    return a -b

def mul(a, b):
    return a * b

def div(a, b):
    if isinstance(b, (int, float)):
        if b >= 0:
            b = max(1e-2, b)
        else:
            b = min(-1e-2, b)
    return a / b

def sqrt(a):
    if isinstance(a, (int, float)):
        if a < 0:
            a = -a
    else:
        a = np.abs(a)
    return np.sqrt(a)

def ssqrt(a):
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

def square(a):
    return a ** 2

def ignore(a):
    return a

def ignore_int(a):
    return a

@numba.njit
def sma(x: np.ndarray, d: int) -> np.ndarray:
    ma = np.zeros((x.shape[0], x.shape[0]-d+1))
    for i in range(ma.shape[0]-d+1):
        ma[i:(i + d), i] = 1/d
    res =  x.T @ ma
    res = np.hstack((np.full((x.shape[1], d-1), fill_value=np.nan), res))
    return res.T

def delay(x: np.ndarray, d: int) -> np.ndarray:
    res = np.roll(x, d, axis=0)
    res[:d, :] = np.nan
    return res

def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    for row in range(data.shape[0] - d + 1):
        rank = data[row:row + d].argsort(axis=0).argsort(axis=0)
        res[row + d - 1, :] = rank[-1, :]
    return res

def ema(x: np.ndarray, d:int):
    ema = np.zeros([x.shape[0], x.shape[0]-d+1])


if __name__ == '__main__':
    np.random.seed(0)
    data = np.array([
        [1, 2, 4, 2, 3],
        [3, 2, 2, 1, 5],
        [2, 4, 3, 1, 5],
        [2, 3, 3, 5, 4],
    ], dtype='float32')
    print(ts_rank(data, 3))

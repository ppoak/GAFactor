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

def sma(x: np.ndarray, d: int):
    ma = np.zeros([x.shape[0], x.shape[0]-d+1])
    for i in range(ma.shape[0]-d+1): # 
        ma[i:i+d,i:i+1] = 1/d
    ma_res =  x.T @ ma
    ma_res = np.c_[ np.zeros([x.shape[1], d-1]) , ma_res]
    return ma_res.T

def ema(x: np.ndarray, d:int):
    ema = np.zeros([x.shape[0], x.shape[0]-d+1])


if __name__ == '__main__':
    data = np.arange(8*9).reshape([8,9])
    print(sma(data, 3))

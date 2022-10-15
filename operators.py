import numpy as np
import pandas as pd


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
        a = a.abs()
    return np.sqrt(a)

def ssqrt(a):
    if isinstance(a, (int, float)):
        if a < 0:
            a = -a
            sign = -1
        else:
            sign = 1
    else:
        sign = pd.DataFrame(np.ones_like(a), index=a.index, columns=a.columns)
        sign = sign.mask(a < 0, -1)
        a = a.abs()
    return sign * np.sqrt(a)

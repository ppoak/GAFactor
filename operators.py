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

import numpy as np


from dual import *


##########
# 配列ユーティリティ関数 array-utility function
#######
def ndim(obj):
    return np.ndim(obj.re)

def shape(obj):
    return np.shape(obj.re)

def size(obj):
    return np.size(obj.re)


##########
# 配列生成関数 array-generate function
#######
def arange(stop, start=None, step=None, dtype=None):
    return Dual(np.arange(stop, start=start, step=step, dtype=dtype), \
                np.arange(stop, start=start, step=step, dtype=dtype))

def zeros(shape, dtype=np.float, order="C"):
    return Dual(np.zeros(shape, order=order), np.zeros(shape, order=order), dtype=dtype)

def zeros_like(a, dtype=None, order="K", subok=True):
    return Dual(np.zeros_like(a, order=order, subok=subok), \
                np.zeros_like(a, order=order, subok=subok), dtype=dtype)

def ones(shape, dtype=np.float, order="C"):
    return Dual(np.ones(shape, order=order), np.ones(shape, order=order), dtype=dtype)

def ones_like(a, dtype=None, order="K", subok=True):
    return Dual(np.ones_like(a, order=order, subok=subok), \
                np.ones_like(a, order=order, subok=subok), dtype=dtype)

def empty(shape, dtype=np.float, order="C"):
    return Dual(np.empty(shape, order=order), np.empty(shape, order=order), dtype=dtype)

def arange(stop, start=None, step=None, dtype=None):
    return Dual(np.arange(start, stop, step), np.arange(start, stop, step), dtype=dtype)

def eye(N, M=None, k=0, dtype=np.float, order="C"):
    return Dual(np.eye(N, M=M, k=k, order=order), np.eye(N, M=M, k=k, order=order), \
                dtype=dtype)

def identity(n, dtype=None):
    return Dual(np.identity(n), np.identity(n), dtype=dtyoe)



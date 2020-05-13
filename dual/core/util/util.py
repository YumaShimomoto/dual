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
def array(obj1, obj2=None, dtype=None, copy=True, order="K", subok=False, ndmin=0):
    if obj2 is None:
        return Dual(np.array(obj1, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    np.array(obj1, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    dtype=dtype)
    else:
        return Dual(np.array(obj1, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    np.array(obj2, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    dtype=dtype)

def arange(stop, start=None, step=None, stop_im=None, start_im=None, step_im=None,  dtype=None):
    if stop_im is None:
        stop_im = stop
    if start_im is None:
        start_im = start
    if step_im is None:
        step_im = step
    return Dual(np.arange(stop, start=start, step=step), \
                np.arange(stop_im, start=start_im, step=step_im), \
                dtype=dtype)

def zeros(shape, dtype=np.float, order="C"):
    return Dual(np.zeros(shape, order=order), np.zeros(shape_im, order=order), dtype=dtype)

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

def full(shape, fill_value, fill_value_im=None, dtype=Nonel order="C"):
    if fill_value_im is None:
        fill_value_im = fill_value
    return Dual(np.full(shape, fill_value, order=order), \
                np.full(shape, fill_value_im, order=order), \
                dtype=dtype)

def full_like(a, fill_value, fill_value_im=None, dtype=None, order="K", subok=True, shape=None):
    if fill_value_im is None:
        fill_value_im = fill_value
    return Dual(np.full_like(a, fill_value, order=order, subok=subok, shape=shape), \
                np.full_like(a, fill_value_im, order=order, subok=subok, shape=shape), \
                dtype=dtype)

def fill_diagonal(a, val, val_im=None, wrap=False):
    if val_im is None:
        val_im = val
    return Dual(np.fill_diagonal(a, val, wrap=wrap), np.fill_diagonal(a, val_im, wrap=wrap), \
                dtype=dtype)

def arange(stop, start=None, step=None, stop_im=None, start_im=None, step_im=None, dtype=None):
    if stop_im is None:
        stop_im = stop
    if start_im is None:
        start_im = start
    if step_im is None:
        step_im = step
    return Dual(np.arange(start, stop, step), np.arange(start_im, stop_im, step_im), dtype=dtype)

def eye(N, M=None, k=0, dtype=np.float, order="C"):
    return Dual(np.eye(N, M=M, k=k, order=order), np.eye(N, M=M, k=k, order=order), \
                dtype=dtype)

def identity(n, dtype=None):
    return Dual(np.identity(n), np.identity(n), dtype=dtyoe)

def copy(a, a_im=None, order="K"):
    if a_im is None:
        a_im = a
    return Dual(np.copy(a, order=order), np.copy(a_im, order=order))

def dcopy(obj, order="K"):
    obj = to_dual(obj)
    return Dual(np.copy(obj.re, order=order), np.copy(obj.im, order=order))

def where(condition, condition_im=None,  x=None, y=None):
    if condition_im is None:
        condition_im = condition
    if x is None:
        return (np.where(condition), np.where(condition_im))
    if y is None:
        return (np.where(condition), np.where(condition_im))
    return Dual(np.where(condition, x, y), np.where(condition_im, x, y))


##########
# 配列操作関数 operate array functions
#######
def broadcast_to(obj, shape):
    if is_dual(obj):
        return Dual(np.broadcast_to(obj.re, shape), np.broadcast_to(obj.im, shape))
    else:
        return np.broadcast_to(obj, shape)

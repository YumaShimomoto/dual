#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *


##########
# 統計関数 statistical functions
#######
def max(obj, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return np.max(obj.re, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


def max_imag(obj, axis=None, out=None, keepdims=np._NoValue, \
              initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return np.max(obj.im, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


def max_dual(obj, axis=None, out=None, keepdims=np._NoValue, \
              initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.max(obj.re, axis=axis, out=out, keepdims=keepdims, \
                        initial=initial, where=where), \
                np.max(obj.im, axis=axis, out=out, keepdims=keepdims, \
                        initial=initial, where=where))


def min(obj, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return np.min(obj.re, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


def min_imag(obj, axis=None, out=None, keepdims=np._NoValue, \
             initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return np.min(obj.im, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


def min_dual(obj, axis=None, out=None, keepdims=np._NoValue, \
             initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.min(obj.re, axis=axis, out=out, keepdims=keepdims, \
                       initial=initial, where=where), \
                np.min(obj.im, axis=axis, out=out, keepdims=keepdims, \
                        initial=initial, where=where))


def nanmax(obj, axis=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.nanmax(obj.re, axis=axis, out=out, keepdims=keepdims)


def nanmax_imag(obj, axis=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.nanmax(obj.im, axis=axis, out=out, keepdims=keepdims)


def nanmax_dual(obj, axis=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.nanmax(obj.re, axis=axis, out=out, keepdims=keepdims), \
                np.nanmax(obj.im, axis=axis, out=out, keepdims=keepdims))


def nanmin(obj, axis=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.nanmin(obj.re, axis=axis, out=out, keepdims=keepdims)


def nanmin_imag(obj, axis=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.nanmin(obj.im, axis=axis, out=out, keepdims=keepdims)


def nanmax_dual(obj, axis=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.nanmin(obj.re, axis=axis, out=out, keepdims=keepdims), \
                np.nanmin(obj.im, axis=axis, out=out, keepdims=keepdims))


def mean(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.mean(obj.re, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def mean_imag(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.mean(obj.im, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def mean_dual(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.mean(obj.re, axis=axis, out=out, keepdims=keepdims), \
                np.mean(obj.im, axis=axis, out=out, keepdims=keepdims), \
                dtype=dtype)


def sum(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue, \
        initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return np.sum(obj.re, axis=axis, dtype=dtype, out=out, keepdims=keepdims, \
                  initial=initial, where=where)


def sum_imag(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue, \
             initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return np.sum(obj.im, axis=axis, dtype=dtype, out=out, keepdims=keepdims, \
                  initial=initial, where=where)


def sum_dual(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue, \
         initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.sum(obj.re, axis=axis, out=out, keepdims=keepdims, \
                       initial=initial, where=where), \
                np.sum(obj.im, axis=axis, out=out, keepdims=keepdims, \
                       initial=initial, where=where))


def median(obj, axis=None, out=None, overwrite_input=False, keepdims=False):
    obj = to_dual(obj)
    return np.median(obj.re, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)


def median_imag(obj, axis=None, out=None, overwrite_input=False, keepdims=False):
    obj = to_dual(obj)
    return np.median(obj.im, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)


def median_dual(obj, axis=None, out=None, overwrite_input=False, keepdims=False):
    obj = to_dual(obj)
    return Dual(np.median(obj.re, axis=axis, out=out, \
                          overwrite_input=overwrite_input, keepdims=keepdims), \
                np.median(obj.im, axis=axis, out=out, \
                          overwrite_input=overwrite_input, keepdims=keepdims))


def std(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.std(obj.re, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def std_imag(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.std(obj.im, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def std_dual(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.std(obj.re, axis=axis, out=out, ddof=ddof, keepdims=keepdims), \
                np.std(obj.im, axia=axis, out=out, ddof=ddof, keepdims=keepdims), \
                dtype=dtype)


def var(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.var(obj.re, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def var_imag(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.var(obj.im, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def var_dual(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return Dual(np.var(obj.re, axis=axis, out=out, ddof=ddof, keepdims=keepdims), \
                np.var(obj.im, axis=axis, out=out, ddof=ddof, keepdims=keepdims), \
                dtype=dtype)


def argmax(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmax(obj.re, axis=axis, out=out)


def argmax_imag(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmax(obj.im, axis=axis, out=out)


def argmax_dual(obj, axis=None, out=None):
    obj = to_dual(obj)
    return {"re": np.argmax(obj.re, axis=axis, out=out), \
            "im": np.argmax(obj.im, axis=axis, out=out)}


def nanargmax(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmax(obj.re, axis=axis)


def nanargmax_imag(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmax(obj.im, axis=axis)

def nanargmax_dual(obj, axis=None):
    obj = to_dual(obj)
    return {"re": np.nanargmax(obj.re, axis=axis), \
            "im": np.nanargmax(obj.im, axis=axis)}


def nanargmin(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmin(obj.re, axis=axis)


def nanargmin_imag(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmin(obj.im, axis=axis)


def nanargmin_dual(obj, axis=None):
    obj = to_dual(obj)
    return {"re": np.nanargmin(obj.re, axis=axis), \
            "im": np.nanargmin(obj.im, axis=axis)}


def argmin(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmax(obj.re, axis=axis, out=out)


def argmin_imag(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmin(obj.im, axis=axis, out=out)


def argmin_dual(obj, axis=None, out=None):
    obj = to_dual(obj)
    return {"re": np.argmin(obj.re, axis=axis, out=out), \
            "im": np.argmin(obj.im, axis=axis, out=out)}

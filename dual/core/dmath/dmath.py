#
# @Author: Yuma Shimomoto (2020)
# @email: yuma_1215@docomo.ne.jp
#

import numpy as np


from ..dual import *


##########
# 数学定数 mathematical constants
#######
pi = 3.141592653589793
e = 2.718281828459045


##########
# 数学関数 mathematical functions
#######
def power(obj, n):
    obj = to_dual(obj)
    return obj ** n

def square(obj):
    obj = to_dual(obj)
    return obj ** 2

def sqrt(obj):
    obj = to_dual(obj)
    return obj ** 0.5

def cbrt(obj):
    obj = to_dual(obj)
    return obj ** (1 / 3)

def exp(obj):
    obj = to_dual(obj)
    return Dual(np.exp(obj.re), np.exp(obj.re) * obj.im)

def exp2(obj):
    obj = to_dual(obj)
    return Dual(np.exp2(obj.re), 2 * obj.re * np.log(obj.re) * obj.im)

def expm1(obj):
    obj = to_dual(obj)
    return exp(obj) - 1

def log(obj):
    obj = to_dual(obj)
    return Dual(np.log(obj.re), obj.im / obj.re)

def logn(obj, n):
    return log(obj) / log(n)

def log2(obj):
    return logn(obj, 2)

def log10(obj):
    return logn(obj, 10)

def log1p(obj):
    return log(1 + obj)

def logaddexp(obj1, obj2):
    return log(exp(obj1) + exp(obj2))

def logaddexp2(obj1, obj2):
    return log2(2 ** obj1 + 2 ** obj2)

def sin(obj):
    obj = to_dual(obj)
    return Dual(np.sin(obj.re), np.cos(obj.re) * obj.im)

def cos(obj):
    obj = to_dual(obj)
    return Dual(np.cos(obj.re), - np.sin(obj.re) * obj.im)

def tan(obj):
    return sin(obj) / cos(obj)

def sinh(obj):
    return 0.5 * (exp(obj) - exp(-obj))

def cosh(obj):
    return 0.5 * (exp(obj) + exp(-obj))

def tanh(obj):
    return sinh(obj) / cosh(obj)

def arcsin(obj):
    obj = to_dual(obj)
    return Dual(np.arcsin(obj.re), obj.im / sqrt(1 - obj.re ** 2))

def arccos(obj):
    obj = to_dual(obj)
    return Dual(np.arccos(obj.re), - obj.im / sqrt(1 - obj.re ** 2))

def arctan(obj):
    obj = to_dual(obj)
    return Dual(np.arctan(obj.re), obj.im / (1 + obj.re ** 2))

def arctan2(obj1, obj2):
    return arctan(obj1 / obj2)

def arcsinh(obj):
    obj = to_dual(obj)
    return Dual(np.arcsinh(obj.re), obj.im / sqrt(1 + obj.re ** 2))

def arccosh(obj):
    obj = to_dual(obj)
    return Dual(np.arccosh(obj.re), obj.im / sqrt(obj.re ** 2 - 1))

def arctanh(obj):
    obj = to_dual(obj)
    return Dual(np.arctanh(obj.re), obj.im / (1 - obj.re ** 2))

def csc(obj):
    return 1 / sin(obj)

def sec(obj):
    return 1 / cos(obj)

def cot(obj):
    return 1 / tan(obj)

def csch(obj):
    return 1 / sinh(obj)

def sech(obj):
    return 1 / cosh(obj)

def coth(obj):
    return 1 / tanh(obj)

def arccsc(obj):
    return arcsin(1 / obj)

def arcsec(obj):
    return arccos(1 / obj)

def arccot(obj):
    return arctan(1 / obj)

def arccsch(obj):
    return arcsinh(1 / obj)

def arcsech(obj):
    return arccosh(1 / obj)

def arccoth(obj):
    return arctanh(1 / obj)

def sinc(obj, norm=False):
    if norm:
        return sin(pi * obj) / (pi * obj)
    else:
        return sin(obj) / obj


##########
# 数学演算関数 mathematical operation function
#######
def add(obj1, obj2):
    return obj1 + obj2

def subtract(obj1, obj2):
    return obj1 - obj2

def multiply(obj1, obj2):
    return obj1 * obj2

def divide(obj1, obj2):
    return obj1 / obj2

def dot(obj1, obj2):
    return obj1 @ obj2

def matmul(obj1, obj2):
    return obj1 @ obj2

def reciprocal(obj):
    return 1 / obj


##########
# 数学に関連するユーティリティ math related utilities
#######
def deg2rad(obj):
    obj = to_dual(obj)
    return obj * pi / 180

def rad2deg(obj):
    obj = to_dual(obj)
    return obj * 180 / pi

def floor(obj):
    obj = to_dual(obj)
    return Dual(np.floor(obj.re), np.floor(obj.im))

def trunc(obj):
    obj = to_dual(obj)
    return Dual(np.trunc(obj.re), np.trunc(obj.im))

def ceil(obj):
    obj = to_dual(obj)
    return Dual(np.ceil(obj.re), np.ceil(obj.im))

def round(obj, decimals=0):
    obj = to_dual(obj)
    return Dual(np.round(obj.re, decimals=decimals), np.round(obj.im, decimals=decimals))

def rint(obj):
    obj = to_dual(obj)
    return Dual(np.rint(obj.re), np.rint(obj.im), dtype=np.int)

def fix(obj):
    obj = to_dual(obj)
    return Dual(np.fix(obj.re), np.fix(obj.im), dtype=np.int)

def real(obj):
    obj = to_dual(obj)
    return obj.re

def imag(obj):
    obj = to_dual(obj)
    return obj.im

def conj(obj):
    obj = to_dual(obj)
    return Dual(obj.re, -obj.im)

def absolute(obj):
    return abs(obj)

def _compare(obj1, obj2, func):
    obj1 = to_dual(obj1)
    obj2 = to_dual(obj2)

    result = Dual(np.zeros(obj1.shape), np.zeros(obj1.shape))
    result.re = func(obj1.re, obj2.re)
    indices = obj1.re == result.re
    result.im[indices] = obj1.im[indices]
    result.im[~indices] = obj2.im[~indices]

    eq_indices = obj1.re == obj2.re
    im_elements = func(obj1.im, obj2.im)
    im_indices = im_elements == obj1.im
    result.re[im_indices * eq_indices] = obj1.re[im_indices * eq_indices]
    result.im[im_indices * eq_indices] = obj1.im[im_indices * eq_indices]
    result.re[~im_indices * eq_indices] = obj2.re[~im_indices * eq_indices]
    result.im[~im_indices * eq_indices] = obj2.im[~im_indices * eq_indices]
    return result

def fmax(obj1, obj2):
    return _compare(obj1, obj2, np.fmax)

def fmin(obj1, obj2):
    return _compare(obj1, obj2, np.fmin)

def maximum(obj1, obj2):
    return _compare(obj1, obj2, np.maximum)

def minimum(obj1, obj2):
    return _compare(obj1, obj2, np.minimum)

def isnan(obj):
    obj = to_dual(obj)
    return np.isnan(obj.re) | np.isnan(obj.im)

def isfinite(obj):
    obj = to_dual(obj)
    return np.isfinite(obj.re) | np.isfinite(obj.im)

def isinf(obj):
    obj = to_dual(obj)
    return np.isinf(obj.re) | np.isinf(obj.im)

def sign(obj):
    obj = to_dual(obj)
    return np.sign(obj.re)

def sign_imag(obj):
    obj = to_dual(obj)
    return np.sign(obj.im)


##########
# 統計関数 statistical functions
#######
def max(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    return np.max(obj.re, axis=axis, out=out, keepdims=keepdims)

def max_imag(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    return np.max(obj.im, axis=axis, out=out, keepdims=keepdims)

def min(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    return np.min(obj.re, axis=axis, out=out, keepdims=keepdims)

def min_imag(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    return np.min(obj.im, axis=axis, out=out, keepdims=keepdims)

def nanmax(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    indices = ~np.isnan(obj.re)
    return max(obj[indices], axis=axis, out=out, keepdims=keepdims)

def nanmax_imag(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    indices = ~np.isnan(obj.im)
    return max_imag(obj[indices], axis=axis, out=out, keepdims=keepdims)

def nanmin(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    indices = ~np.isnan(obj.re)
    return min(obj[indices], axis=axis, out=out, keepdims=keepdims)

def nanmin_imag(obj, axis=None, out=None, keepdims=False):
    obj = to_dual(obj)
    indices = ~np.isnan(obj.im)
    return min_imag(obj[indices], axis=axis, out=out, keepdims=keepdims)

def mean(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.mean(obj.re, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def mean_imag(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.mean(obj.im, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def dmean(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    obj = to_dual(obj)
    return Dual(mean(obj), mean_imag(obj))

def sum(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue, \
        initial=np._NoValue, where=np._NoValue):
    obj = to_dual(obj)
    return np.sum(obj.re, axis=axis, dtype=dtype, out=out, keepdims=keepdims, \
                  initial=initial, where=where)

def sum_imag(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue, \
             initial=np._NoValue, where=no._NoValue):
    obj = to_dual(obj)
    return np.sum(obj.im, axis=axis, dtype=dtype, out=out, keepdims=keepdims, \
                  initial=initial, where=where)

def dsum(obj, axis=None, dtype=None, out=None, keepdims=np._NoValue, \
         initial=np._NoValue, where=np._NoValue)
    obj = to_dual(obj)
    return Dual(sum(obj), sum_imag(obj))

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    obj = to_dual(obj)
    return np.median(obj.re, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)

def median_imag(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    obj = to_dual(obj)
    return np.median(obj.im, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)

def std(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.std(obj.re, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

def std_imag(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.std(obj.im, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

def var(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.var(obj.re, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

def var_imag(obj, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    obj = to_dual(obj)
    return np.var(obj.im, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

def argmax(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmax(obj.re, axis=axis, out=out)

def argmax_imag(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmax(obj.im, axis=axis, out=out)

def nanargmax(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmax(obj.re, axis=axis)

def nanargmax_imag(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmax(obj.im, axis=axis)

def nanargmin(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmin(obj.re, axis=axis)

def nanargmin_imag(obj, axis=None):
    obj = to_dual(obj)
    return np.nanargmin(obj.im, axis=axis)

def argmin(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmax(obj.re, axis=axis, out=out)

def argmin_imag(obj, axis=None, out=None):
    obj = to_dual(obj)
    return np.argmin(obj.im, axis=axis, out=out)

#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *
from .exponential import *


def log(obj):
    obj = to_dual(obj)
    return Dual(np.log(obj.re), obj.im / obj.re)


def logn(obj, n):
    return log(obj) / log(n)


def log2(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.log2(obj.re), obj.im / (obj.re * np.log(2)))
    return logn(obj, 2)


def log10(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.log10(obj.re), obj.im / (obj.re * np.log(10)))
    return logn(obj, 10)


def log1p(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.log1p(obj.re), obj.im / (1 + obj.re))
    return log(1 + obj)


def logaddexp(obj1, obj2):
    return log(exp(obj1) + exp(obj2))


def logaddexp2(obj1, obj2):
    return log2(2 ** obj1 + 2 ** obj2)

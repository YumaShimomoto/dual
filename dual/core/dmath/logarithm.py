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
    obj = to_dual(obj)
    return Dual(np.log(obj.re) / np.log(n), obj.im / (np.log(n) * obj.re))


def log2(obj):
    obj = to_dual(obj)
    return Dual(np.log2(obj.re), obj.im / (obj.re * np.log(2)))


def log10(obj):
    obj = to_dual(obj)
    return Dual(np.log10(obj.re), obj.im / (obj.re * np.log(10)))


def log1p(obj):
    obj = to_dual(obj)
    return Dual(np.log1p(obj.re), obj.im / (1 + obj.re))


def logaddexp(obj1, obj2):
    obj1, obj2 = to_dual(obj1), to_dual(obj2)
    return Dual(np.logaddexp(obj1.re, obj2.re),
                (obj1.im * np.exp(obj1.re)
                 + obj2.im * np.exp(obj2.re))/(np.exp(obj1.re)
                 + np.exp(obj2.re)))
    #return log(exp(obj1) + exp(obj2))


def logaddexp2(obj1, obj2):
    obj1, obj2 = to_dual(obj1), to_dual(obj2)
    return Dual(np.logaddexp2(obj1.re, obj2.re),
                (obj1.im * 2 ** obj1.re + obj2.im * 2 ** obj2.re)
                 /(2 ** obj1.re + 2 ** obj2.re))
    #return log2(2 ** obj1 + 2 ** obj2)

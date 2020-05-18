#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *
from .const import *
from .exponential import *


##########
# 三角関数 trigonometrical functions
#######
def sin(obj):
    obj = to_dual(obj)
    return Dual(np.sin(obj.re), np.cos(obj.re) * obj.im)


def cos(obj):
    obj = to_dual(obj)
    return Dual(np.cos(obj.re), - np.sin(obj.re) * obj.im)


def tan(obj):
    return sin(obj) / cos(obj)


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


def csc(obj):
    return 1 / sin(obj)


def sec(obj):
    return 1 / cos(obj)


def cot(obj):
    return 1 / tan(obj)


def arccsc(obj):
    return arcsin(1 / obj)


def arcsec(obj):
    return arccos(1 / obj)


def arccot(obj):
    return arctan(1 / obj)


def sinc(obj, norm=False):
    if norm:
        return sin(pi * obj) / (pi * obj)
    else:
        return sin(obj) / obj

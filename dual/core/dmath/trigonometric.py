#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *
from .const import pi


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
    obj = to_dual(obj)
    return Dual(np.tan(obj.re), obj.im / np.cos(obj.re) ** 2)


def arcsin(obj):
    obj = to_dual(obj)
    return Dual(np.arcsin(obj.re), obj.im / np.sqrt(1 - obj.re ** 2))


def arccos(obj):
    obj = to_dual(obj)
    return Dual(np.arccos(obj.re), - obj.im / np.sqrt(1 - obj.re ** 2))


def arctan(obj):
    obj = to_dual(obj)
    return Dual(np.arctan(obj.re), obj.im / (1 + obj.re ** 2))


def arctan2(obj1, obj2):
    return arctan(obj1 / obj2)


def csc(obj):
    obj = to_dual(obj)
    return Dual(1 / np.sin(obj.re), - obj.im / (np.tan(obj.re) * np.sin(obj.re)))


def sec(obj):
    obj = to_dual(obj)
    return Dual(1 / np.cos(obj.re), obj.im * np.tan(obj.re) / np.cos(obj.re))


def cot(obj):
    obj = to_dual(obj)
    return Dual(1 / np.tan(obj.re), - obj.im / np.sin(obj.re) ** 2)


def arccsc(obj):
    obj = to_dual(obj)
    return Dual(np.arcsin(1 / obj.re), - obj.im / (obj.re ** 2 * np.sqrt(1 - 1 / obj.re ** 2)))


def arcsec(obj):
    obj = to_dual(obj)
    return Dual(np.arccos(1 / obj.re), obj.im / (obj.re ** 2 * np.sqrt(1 - 1 / obj.re ** 2)))


def arccot(obj):
    obj = to_dual(obj)
    return Dual(np.arctan(1 / obj.re), - obj.im / (1 + obj.re ** 2))


def sinc(obj, norm=True):
    if norm:
        obj = to_dual(obj)
        return Dual(np.sinc(obj.re), \
                    obj.im * (obj.re * np.cos(obj.re * pi) \
                              - np.sin(obj.re * pi) / pi) / obj.re ** 2)
    else:
        obj = to_dual(obj)
        return Dual(np.sinc(obj.re / pi), \
                    obj.im * (obj.re * np.cos(obj.re) - np.sin(obj.re)) / obj.re ** 2)

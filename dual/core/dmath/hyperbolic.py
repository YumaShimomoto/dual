
#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *
from .const import pi
from .exponential import sqrt


##########
# 双曲線関数 hyperbolical functions
#######
def sinh(obj):
    obj = to_dual(obj)
    return Dual(np.sinh(obj.re), obj.im * np.cosh(obj.re))


def cosh(obj):
    obj = to_dual(obj)
    return Dual(np.cosh(obj.re), obj.im * np.sinh(obj.re))


def tanh(obj):
    obj = to_dual(obj)
    return Dual(np.tanh(obj.re), obj.im / np.cosh(obj.re) ** 2)


def arcsinh(obj):
    obj = to_dual(obj)
    return Dual(np.arcsinh(obj.re), obj.im / sqrt(1 + obj.re ** 2))


def arccosh(obj):
    obj = to_dual(obj)
    return Dual(np.arccosh(obj.re), obj.im / sqrt(obj.re ** 2 - 1))


def arctanh(obj):
    obj = to_dual(obj)
    return Dual(np.arctanh(obj.re), obj.im / (1 - obj.re ** 2))


def csch(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.csch(obj.re), - obj.im * np.coth(obj.re) * np.csch(obj.re))
    return 1 / sinh(obj)


def sech(obj, switch=True)
    if switch:
        obj = to_dual(obj)
        return Dual(np.sech(obj.re), - obj.im * np.tanh(obj.re) * np.sech(obj.re))
    return 1 / cosh(obj)


def coth(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.coth(obj.re), - obj.im / np.sinh(obj.re) ** 2)
    return 1 / tanh(obj)


def arccsch(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.arccsch(obj.re), - obj.im / (obj.re ** 2 * sqrt(1 + 1 / obj.re ** 2)))
    return arcsinh(1 / obj)


def arcsech(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.arcsech(obj.re), \
                    - obj.im / (obj.re * (obj.re + 1) * sqrt((1 - obj.re) / (1 + obj.re))))
    return arccosh(1 / obj)


def arccoth(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(np.arccoth(obj.re), obj.im / (1 - obj.re ** 2))
    return arctanh(1 / obj)


def sinch(obj, norm=False):
    if norm:
        return sinh(pi * obj) / (pi * obj)
    else:
        return sinh(obj) / obj

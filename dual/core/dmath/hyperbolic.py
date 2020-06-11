
#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *
from .const import pi


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
    return Dual(np.arcsinh(obj.re), obj.im / np.sqrt(1 + obj.re ** 2))


def arccosh(obj):
    obj = to_dual(obj)
    return Dual(np.arccosh(obj.re), obj.im / np.sqrt(obj.re ** 2 - 1))


def arctanh(obj):
    obj = to_dual(obj)
    return Dual(np.arctanh(obj.re), obj.im / (1 - obj.re ** 2))


def csch(obj, switch=True):
    if switch:
        obj = to_dual(obj)
        return Dual(1 / np.sinh(obj.re),
                    - obj.im / (np.tanh(obj.re) * np.sinh(obj.re)))
    return 1 / sinh(obj)


def sech(obj, s=True):
    if s:
        obj = to_dual(obj)
        return Dual(1 / np.cosh(obj.re),
                    - obj.im * np.tanh(obj.re) / np.cosh(obj.re))
    return 1 / cosh(obj)


def coth(obj, s=True):
    if s:
        obj = to_dual(obj)
        return Dual(1 / np.tanh(obj.re),
                    - obj.im / np.sinh(obj.re) ** 2)
    return 1 / tanh(obj)


def arccsch(obj, s=True):
    if s:
        obj = to_dual(obj)
        return Dual(np.arcsinh(1 / obj.re),
                    - obj.im/(obj.re ** 2 * np.sqrt(1 + 1 / obj.re ** 2)))
    return arcsinh(1 / obj)


def arcsech(obj, s=True):
    obj = to_dual(obj)
    return Dual(np.arccosh(1 / obj.re),
                - obj.im / (obj.re ** 2 * np.sqrt(1 / obj.re - 1)
                         * np.sqrt(1 / obj.re + 1)))


def arccoth(obj):
    obj = to_dual(obj)
    return Dual(np.arctanh(1 / obj.re), obj.im / (1 - obj.re ** 2))


def sinch(obj, norm=True):
    if norm:
        obj = to_dual(obj)
        return Dual(np.sinh(obj.re * pi) / (obj.re * pi),
                    obj.im * (obj.re * np.cosh(obj.re * pi)
                              - np.sinh(obj.re * pi) / pi) / obj.re ** 2)
    else:
        obj = to_dual(obj)
        return Dual(np.sinh(obj.re) / obj.re,
                    obj.im * (obj.re * np.cosh(obj.re)
                              - np.sinh(obj.re)) / obj.re ** 2)

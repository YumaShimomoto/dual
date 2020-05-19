
#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *


##########
# 指数関数 exponential functions
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
    return Dual(np.exp(obj.re), obj.im * np.exp(obj.re))


def exp2(obj):
    obj = to_dual(obj)
    return Dual(np.exp2(obj.re), obj.im * 2 ** obj.re * np.log(2))


def expm1(obj):
    obj = to_dual(obj)
    return Dual(np.expm1(obj.re), obj.im * np.exp(obj.re))

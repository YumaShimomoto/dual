#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


from ..dual import *
from .const import *


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
    return Dual(np.round(obj.re, decimals=decimals),
                np.round(obj.im, decimals=decimals))


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
    return Dual(obj.re, - obj.im)


def absolute(obj):
    return abs(obj)

def _compare(obj1, obj2, func):
    obj1, obj2 = to_dual(obj1), to_dual(obj2)

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


def sign_dual(obj):
    obj = to_dual(obj)
    return Dual(np.sign(obj.re), np.sign(obj.im))


def signbit(obj):
    obj = to_dual(obj)
    return np.signbit(obj.re)


def signbit_imag(obj):
    obj = to_dual(obj)
    return np.signbit(obj.im)


def signbit_dual(obj):
    obj = to_dual(obj)
    return {"re": np.signbit(obj.re), \
            "im": np.signbit(obj.im)}


def copysign(obj1, obj2):
    obj1, obj2 = to_dual(obj1), to_dual(obj2)
    return Dual(np.copysign(obj1.re, obj2.re),
                np.copysign(obj1.im, obj2.im))

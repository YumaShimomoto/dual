
#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import numpy as np


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


def inner(obj1, obj2):
    obj1, obj2 = to_dual(obj1), to_dual(obj2)
    return Dual(np.inner(obj1.re, obj2.re), np.inner(obj1.im, obj2.im))

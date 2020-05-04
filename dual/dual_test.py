#
# @Author: Yuma Shimomoto (2020)
# @email: yuma_1215@docomo.ne.jp
#
"""
参考：https://github.com/tmurakami1234/my_python_module/blob/master/dual/dual.py
"""

import numpy as np


from dual import Dual


def test():
    print("+---------------------+")
    print("| Test of Dual module |")
    print("+---------------------+\n")

    print("__init__のテスト", "test __init__", "\n")
    x = Dual(1, 2)
    y = Dual([1, 2], [3, 4])
    z = Dual([1, 2], 3)
    w = Dual(y, z)
    v = Dual(1, 0)
    print("--期待結果--", "--expected--")
    print("x = Dual(1, 2)")
    print("y = Dual([1, 2], [3, 4])")
    print("z = Dual([1, 2], [3, 3])")
    print("w = Dual([1, 2], [4, 6])")
    print("v = Dual(1, 0)")
    print("--結果--", "--result--")
    print("x = {}".format(x))
    print("y = {}".format(y))
    print("z = {}".format(z))
    print("w = {}".format(w))
    print("v = {}".format(v))
    print("-------------------------------------------\n")

    print("比較演算子のテスト", "test comparison operators", "\n")
    print("--期待結果--", "--expected--")
    print("y < z = [False, False]")
    print("y <= z = [True, False]")
    print("y == z = [True, False]")
    print("y != z = [False, True]")
    print("y > z = [False, True]")
    print("y >= z = [True, True]")
    print("x < v = [False, False]")
    print("x <= v = [True, False]")
    print("x == v = [True, False]")
    print("x != v = [False, True]")
    print("x > v = [False, True]")
    print("x >= v = [True, True]")
    print("--結果--", "--result--")
    print("y < z = {}".format(y < z))
    print("y <= z = {}".format(y <= z))
    print("y == z = {}".format(y == z))
    print("y != z = {}".format(y != z))
    print("y > z = {}".format(y > z))
    print("y >= z = {}".format(y >= z))
    print("x < v = {}".format(x < v))
    print("x <= v = {}".format(x <= v))
    print("x == v = {}".format(x == v))
    print("x != v = {}".format(x != v))
    print("x > v = {}".format(x > v))
    print("x >= v = {}".format(x >= v))
    print("-------------------------------------------\n")

    print("__hash__のテスト", "test __hash__", "\n")
    print("--期待結果--", "--expected--")
    print("hash(x) = {}".format(hash((float(x.re), float(x.im)))))
    print("hash(y) = {}".format(hash((y.re[0], y.im[0])) + hash((y.re[1], y.im[1]))))
    print("hash(z) = {}".format(hash((z.re[0], z.im[0])) + hash((z.re[1], z.im[1]))))
    print("hash(w) = {}".format(hash((w.re[0], w.im[0])) + hash((w.re[1], w.im[1]))))
    print("hash(v) = {}".format(hash(float(v.re))))
    print("--結果--", "--result--")
    print("hash(x) = {}".format(hash(x)))
    print("hash(y) = {}".format(hash(y)))
    print("hash(z) = {}".format(hash(z)))
    print("hash(w) = {}".format(hash(w)))
    print("hash(v) = {}".format(hash(v)))
    print("-------------------------------------------\n")

    print("__bool__と__nonzero__のテスト", "test __bool__ and __nonzero__", "\n")
    print("--期待結果--", "--expected--")
    print("bool(x) = True")
    print("bool(y) = True")
    print("bool(Dual()) = False")
    print("--結果--", "--result--")
    print("bool(x) = {}".format(bool(x)))
    print("bool(y) = {}".format(bool(y)))
    print("bool(Dual()) = {}".format(bool(Dual())))
    print("-------------------------------------------\n")

    print("__setarry__のテスト", "test __setattr__", "\n")
    print("--期待動作--", "--expected--")
    print("x.re = 1", "成功", "succeed")
    print("x.im = 2", "成功", "succeed")
    print("x.hoge = 3", "失敗", "fail")
    print("--結果--", "--result--")
    try:
        x.re = 1
        print("x.re = 1", "成功", "succeed")
    except:
        print("x.re = 1", "失敗", "fail")
    try:
        x.im = 2
        print("x.im = 2", "成功", "succeed")
    except:
        print("x.im = 2", "失敗", "fail")
    try:
        x.hoge = 3
        print("x.hoge = 3", "成功", "succeed")
    except:
        print("x.hoge = 3", "失敗", "fail")
    print("-------------------------------------------\n")

    print("__len__のテスト", "test __len__", "\n")
    print("--期待結果--", "--expected--")
    print("len(x) = 1")
    print("len(y) = 2")
    print("--結果--", "--result--")
    print("len(x) = {}".format(len(x)))
    print("len(y) = {}".format(len(y)))
    print("-------------------------------------------\n")

    print("__getitem__と__setitem__、__delitem__のテスト", "test __getitem__,  __setitem__ and __delitem__", "\n")
    print("--期待結果--", "--expected--")
    print("x[0] = Dual(1, 2)")
    print("y[1] = Dual(2, 4)")
    # スライスについて about slice object
    # 参考(reference)：https://qiita.com/tanuk1647/items/276d2be36f5abb8ea52e
    print("y[0 : 2] = Dual([1, 3], [2, 4])")
    print("z[2]", "失敗", "fail")
    print("y[0] = Dual(0, 0)", "成功", "succeed")
    print("del y[1] = Dual(0, 0)", "成功", "succeed")
    print("y[0 : 2] = Dual([1, 2], [3, 4])", "成功", "succeed")
    print("y[2] = Dual(0, 0)", "失敗", "fail")
    print("del y[2] = Dual(0, 0)", "失敗", "fail")
    print("--結果--", "--result--")
    print("x[0] = {}".format(x[0]))
    print("y[1] = {}".format(y[1]))
    print("y[0 : 2] = {}".format(y[0 : 2]))
    try:
        print("z[2] = {}".format(z[2]))
    except:
        print("z[2]", "失敗", "fail")
    try:
        y[0] = Dual(0, 0)
        print("y[0] = {}".format(y[0]), "成功", "succeed")
    except:
        print("y[0] = {}".format(y[0]), "失敗", "fail")
    try:
        del y[1]
        print("del y[1] = {}".format(y[1]), "成功", "succeed")
    except:
        print("del y[1] = {}".format(y[1]), "失敗", "fail")
    try:
        y[0 : 2] = Dual([1, 2], [3, 4])
        print("y[0 : 2] = {}".format(y[0 : 2]), "成功", "succeed")
    except:
        print("y[0 : 2] = {}".format(y[0 : 2]), "失敗", "fail")
    try:
        y[2] = Dual(0, 0)
        print("y[2] = {}".format(y[2]), "成功", "succeed")
    except:
        print("y[2] = Dual(0, 0)", "失敗", "fail")
    try:
        del y[2]
        print("del y[2] = {}".format(y[2]), "成功", "succeed")
    except:
        print("del y[2] = Dual(0, 0)", "失敗", "fail")
    print("-------------------------------------------\n")

    print("イテレータのテスト", "test iterators", "\n")
    print("--期待結果--", "--expected--")
    print("y[0] = Dual(1, 3)")
    print("y[1] = Dual(2, 4)")
    print("z[1] = Dual(2, 3)")
    print("z[0] = Dual(1, 3)")
    print("w[1] = Dual(2, 6)")
    print("w[0] = Dual(1, 4)")
    print("--結果--", "--result--")
    i = 0
    for a in y:
        print("y[{}] = {}".format(i, a))
        i += 1
    for a in reversed(z):
        print("z[{}] = {}".format(i, a))
        i -= 1
    i = len(w) - 1
    for a in w[::-1]:
        print("w[{}] = {}".format(i, a))
        i -= 1
    print("-------------------------------------------\n")

    print("__contain__のテスト", "test __contain__", "\n")
    print("--期待結果--", "--result--")
    print("Dual(1, 2) in x = True")
    print("Dual(0, 0) in x = False")
    print("Dual(1, 3) in y = True")
    print("Dual(0, 0) in y = False")
    print("--結果--", "--result--")
    print("Dual(1, 2) in x = {}".format(Dual(1, 2) in x))
    print("Dual(0, 0) in x = {}".format(Dual(0, 0) in x))
    print("Dual(1, 3) in y = {}".format(Dual(1, 3) in y))
    print("Dual(0, 0) in y = {}".format(Dual(0, 0) in y))
    print("-------------------------------------------\n")

    print("足し算と引き算のテスト", "test add and substruct operators", "\n")
    print("--期待結果--", "--result--")
    print("x + x = Dual(2, 4)")
    print("x + y = Dual([2, 3], [5, 6])")
    print("y + z = Dual([2, 4], [6, 7])")
    print("x - x = Dual(0, 0)")
    print("x - y = Dual([0, -1], [-1, -2])")
    print("y - z = Dual([0, 0], [0, 1])")
    print("y += z = Dual([2, 4], [6, 7])")
    print("y -= z = Dual([1, 2], [3, 4])")
    print("--結果--", "--result--")
    print("x + x = {}".format(x + x))
    print("x + y = {}".format(x + y))
    print("y + z = {}".format(y + z))
    print("x - x = {}".format(x - x))
    print("x - y = {}".format(x - y))
    print("y - z = {}".format(y - z))
    y += z
    print("y += z = {}".format(y))
    y -= z
    print("y -= z = {}".format(y))
    print("-------------------------------------------\n")

    print("掛け算と行列積のテスト", "test multiplication and matrix product", "\n")
    print("--期待結果--", "--expected--")
    print("x * x = Dual(1, 4)")
    print("x * y = Dual([1, 2], [5, 8])")
    print("y * z = Dual([1, 4], [6, 14])")
    print("z *= y = Dual([1, 4], [6, 14]")
    print("x @ x = Dual(1, 4)")
    print("x @ y = Dual([1, 2], [5, 8])")
    print("y @ x", "失敗", "fail")
    print("y @ w", "失敗", "fail")
    print("y.T @ w = Dual([[1, 2], [2, 4]], [[7, 12], [12, 20]])")
    print("y @ w.T = Dual(5, 27)")
    print("y.T @ w @ x", "失敗", "fail")
    print("y.T @ w @ y", "失敗", "fail")
    print("y.T @ w @ y.T = Dual([[5, 10], [42, 74]])")
    print("(y.T @ w) @ (y.T @ w) = Dual([[5, 10], [10, 20], [[62, 114], [114, 208]])")
    print("--結果--", "--result--")
    print("x * x = {}".format(x * x))
    print("x * y = {}".format(x * y))
    print("y * z = {}".format(y * z))
    z *= y
    print("z *= y = {}".format(z))
    print("x @ x = {}".format(x @ x))
    print("x @ y = {}".format(x @ y))
    try:
        print("y @ x = {}".format(y @ x), "成功", "succeed")
    except:
        print("y @ x", "失敗", "fail")
    try:
        print("y @ w = {}".format(y @ w), "成功", "succeed")
    except:
        print("y @ w", "失敗", "fail")
    print("y.T @ w = {}".format(y.T @ w))
    print("y @ w.T = {}".format(y @ w.T))
    try:
        print("y.T @ w @ x = {}".format(y.T @ w @ x), "成功", "succeed")
    except:
        print("y.T @ w @ x", "失敗", "fail")
    try:
        print("y.T @ w @ y = {}".format(y.T @ w @ y), "成功", "succeed")
    except:
        print("y.T @ w @ y.T", "失敗", "fail")
    print("y.T @ w @ y.T = {}".format(y.T @ w @ y.T))
    print("(y.T @ w) @ (y.T @ w) = {}".format((y.T @ w) @ (y.T @ w)))
    print("-------------------------------------------\n")

    print("割り算と剰余・余りのテスト", "test division, floordiv and mod", "\n")
    print("--期待結果--", "--expected--")
    print("x / x = Dual(1, 0)")
    print("x / y = Dual([1, 0.5], [-1, 0])")
    print("y / w = Dual([1, 1], [-1, -1])")
    print("z /= y = Dual([1, 2], [3, 3])")
    print("x // y", "失敗", "fail")
    print("x % y", "失敗", "fail")
    print("divmod(x, y)", "失敗", "fail")
    print("--結果--", "--result--")
    print("x / x = {}".format(x / x))
    print("x / y = {}".format(x / y))
    print("y / w = {}".format(y / w))
    z /= y
    print("z /= y = {}".format(z))
    try:
        print("x // y = {}".format(x // y), "成功", "succeed")
    except:
        print("x // y", "失敗", "fail")
    try:
        print("x % y = {}".format(x % y), "成功", "succeed")
    except:
        print("x % y", "失敗", "fail")
    try:
        print("divmod(x, y) = {}".format(divmod(x, y)), "成功", "succeed")
    except:
        print("divmod(x, y)", "失敗", "fail")
    print("-------------------------------------------\n")

    print("べき乗のテスト", "test exponentiation")
    print("--期待結果--", "--expected--")
    print("x ** 2 = Dual(1, 4)")
    print("y ** 3 = Dual([1, 8], [9, 48])")
    print("x ** y", "失敗", "fail")
    print("y ** Dual(2, 0) = Dual([1, 4], [6, 16])")
    print("v ** y = Dual([1, 1], [0, 0])")
    print("y ** z", "失敗", "fail")
    print("y ** Dual([1, 2], [0, 0]) = Dual([1, 4], [3, 16])")
    print("--結果--", "--result--")
    print("x ** 2 = {}".format(x ** 2))
    print("y ** 3 = {}".format(y ** 3))
    try:
        print("x ** y = {}".format(x ** y), "成功", "succeed")
    except:
        print("x ** y", "失敗", "fail")
    print("y ** Dual(2, 0) = {}".format(y ** Dual(2, 0)))
    print("v ** y = {}".format(v ** y))
    try:
        print("y ** z = {}".format(y ** z), "成功", "succeed")
    except:
        print("y ** z", "失敗", "fail")
    print("y ** Dual([1, 2], [0, 0]) = {}".format(y ** Dual([1, 2], [0, 0])))
    print("-------------------------------------------\n")

    print("シフト演算と論理演算のテスト", "test shift and logical operation", "\n")
    print("--期待結果--", "--expected--")
    print("x << 1", "失敗", "fail")
    print("x >> 1", "失敗", "fail")
    print("x & x", "失敗", "fail")
    print("x ^ x", "失敗", "fail")
    print("x | x", "失敗", "fail")
    print("--結果--", "--result--")
    try:
        print("x << 1 = {}".format(x << 1), "成功", "succeed")
    except:
        print("x << 1", "失敗", "fail")
    try:
        print("x >> 1 = {}".format(x >> 1), "成功", "succeed")
    except:
        print("x >> 1", "失敗", "fail")
    try:
        print("x & x = {}".format(x & x), "成功", "succeed")
    except:
        print("x & x", "失敗", "fail")
    try:
        print("x ^ x = {}".format(x ^ x), "成功", "succeed")
    except:
        print("x ^ x", "失敗", "fail")
    try:
        print("x | x = {}".format(x | x), "成功", "succeed")
    except:
        print("x | x", "失敗", "fail")
    print("-------------------------------------------\n")

    print("__neg__, __pos__, __abs__のテスト", "test __neg__, __pos__, __abs__", "\n")
    print("--期待結果--", "--expected--")
    print("-x = Dual(-1, -2)")
    print("-y = Dual([-1, -2], [-3, -4])")
    print("+x = Dual(1, 2)")
    print("+y = Dual([1, 2], [3, 4])")
    print("abs(x) = {}".format(np.sqrt(5)))
    print("abs(y) = {}".format(np.sqrt([10, 20])))
    print("--結果--", "--result--")
    print("-x = {}".format(-x))
    print("-y = {}".format(-y))
    print("+x = {}".format(+x))
    print("+y = {}".format(+y))
    print("abs(x) = {}".format(abs(x)))
    print("abs(y) = {}".format(abs(y)))
    print("-------------------------------------------\n")

    print("__complex__, __int__, __float__のテスト", "test __complex__, __int__,  __float__", "\n")
    print("--期待結果--", "--expected--")
    print("complex(x) = 1 + 2j")
    print("complex(y)", "失敗", "fail")
    print("x.complex = 1 + 2j")
    print("y.complex = [1 + 3j, 2 + 4j]")
    print("int(x)", "失敗", "fail")
    print("int(y)", "失敗", "fail")
    print("int(v) = 1")
    print("x.int = Dual(1, 2)")
    print("y.int = Dual([1, 2], [3, 4])")
    print("v.int = 1")
    print("float(x)", "失敗", "fail")
    print("float(y)", "失敗", "fail")
    print("float(v) = 1.")
    print("x.float = Dual([1., 2.])")
    print("y.float = Dual([1., 2.], [3., 4.])")
    print("v.float = 1.")
    print("--結果--", "--result--")
    print("complex(x) = {}".format(complex(x)))
    try:
        print("complex(y) = {}".format(complex(y)), "成功", "succeed")
    except:
        print("complex(y)", "失敗", "fail")
    print("x.complex = {}".format(x.complex))
    print("y.complex = {}".format(y.complex))
    try:
        print("int(x) = {}".format(int(x)), "成功", "succeed")
    except:
        print("int(x)", "失敗", "fail")
    try:
        print("int(y) = {}".format(int(y)), "成功", "succeed")
    except:
        print("int(y)", "失敗", "fail")
    print("int(v) = {}".format(int(v)))
    print("x.int = {}".format(x.int))
    print("y.int = {}".format(y.int))
    print("v.int = {}".format(v.int))
    try:
        print("float(x) = {}".format(float(x)), "成功", "succeed")
    except:
        print("float(x)", "失敗", "fail")
    try:
        print("float(y) = {}".format(float(y)), "成功", "succeed")
    except:
        print("float(y)", "失敗", "fail")
    print("float(v) = {}".format(float(v)))
    print("x.float = {}".format(x.float))
    print("y.float = {}".format(y.float))
    print("v.float = {}".format(v.float))
    print("-------------------------------------------\n")


if __name__ == "__main__":
    test()

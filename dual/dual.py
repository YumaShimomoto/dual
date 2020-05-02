#
# @Author: Yuma Shimomoto (2020)
# @email: yuma_1215@docomo.ne.jp
#
"""
参考：https://github.com/tmurakami1234/my_python_module/blob/master/dual/dual.py
      https://www.pythonprogramming.in/example-of-reversed-magic-method-in-python.html
"""

import numpy as np


class Dual():
    def __init__(self, re=0, im=0):
        # 入力整形
        # Format input
        if is_dual(re):
            if isinstance(re.re, np.ndarray):
                # 引数reが「Dual型である」かつ「ndarray型である」
                # Argument 're' is "Type is 'Dual'" and "Type is 'ndarray'"
                re_re = re.re
                re_im = re.im
            else:
                re_re = np.array(re.re)
                re_im = np.array(re.im)
        else:
            if isinstance(re, np.ndarray):
                # 引数reが「Dual型ではない」かつ「ndarray型である」
                # Argument 're' is "Type isn't 'Dual'" and "Type is 'ndarray'"
                re_re = re
                re_im = np.zeros(re.shape)
            else:
                re_re = np.array(re)
                re_im = np.zeros(re_re.shape)
        if is_dual(im):
            if isinstance(im.re, np.ndarray):
                # 引数imが「Dual型である」かつ「ndarray型である」
                # Argument 'im' is "Type is 'Dual'" and "Type is 'ndarray'"
                im_re = im.re
                im_im = im.im
            else:
                im_re = np.array(im.re)
                im_im = np.array(im.im)
        else:
            if isinstance(im, np.ndarray):
                # 引数imが「Dual型ではない」かつ「ndarray型である」
                # Argument 'im' is "Type isn't 'Dual'" and "Type is 'ndarray'"
                im_re = im
                im_im = np.zeros(im.shape)
            else:
                im_re = np.array(im)
                im_im = np.zeros(im_re.shape)

        # re = x + ex', im = y + ey' の時   # When "re = ~~, im = ~~"
        # Dual(re, im) = re + eim
        #              = (x + ex') + e(y + ey')
        #              = x + e(x' + y')
        _re = np.array(re_re, dtype=np.float)
        _im = np.array(re_im + im_re, dtype=np.float)

        self.__dict__["re"] = _re
        self.__dict__["im"] = _im
        self.__dict__["_i"] = 0

    def __repr__(self):
        if np.ndim(self.re) <= 1:
            return "Dual({}, {})".format(self.re, self.im)
        else:
            return "Dual(\n{}, \n{})".format(self.re, self.im)

    def __str__(self):
        if np.ndim(self.re) <= 1:
            return "Dual({}, {})".format(self.re, self.im)
        else:
            return "Dual(\n{}, \n{})".format(self.re, self.im)

    def __lt__(self, other):
        """
        「実部を比較=>虚部を比較」(辞書順)で比較しています。
        ノルム順にする場合はコメントアウトしているものを使ってください。
        Compared by 'compare real => compare imaginary' (dictionary-order).
        If use norm-order, activate commentouted one.
        """
        other = to_dual(other)
        return (self.re < other.re) | ((self.re == other.re) & (self.im < other.im))

#    def __lt__(self, other):
#        """
#        ノルム順で比較しています。
#        Compared by norm-order.
#        """
#        other = to_dual(other)
#        return self.__abs__() < other.__abs__()

    def __le__(self, other):
        return self.__lt__(other) | self.__eq__(other)

    def __eq__(self, other):
        other = to_dual(other)
        return (self.re == other.re) & (self.im == other.im)

    def __ne__(self, other):
        return ~ self.__eq__(other)

    def __gt__(self, other):
        return ~ self.__le__(other)

    def __ge__(self, other):
        return ~ self.__lt__(other)

    def __hash__(self):
        if len(self) == 1:
            if self.im:
                return hash((float(self.re), float(self.im)))
            else:
                return hash(float(self.re))
        else:
            if np.all(self.im == 0):
                return int(np.sum([hash(self.re[i]) for i in range(len(self))]))
            else:
                return int(np.sum([hash((self.re[i], self.im[i])) for i in range(len(self))]))

    def __bool__(self):
        return bool(np.any(self.__nonzero__()))

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise TypeError("Dual number can't add attribute.")
        else:
            self.__dict__[name] = value

    def __len__(self):
        try:
            return len(self.re)
        except:
            return 1

    def __getitem__(self, key):
        try:
            return Dual(self.re[key], self.im[key])
        except:
            if self.__len__() == 1:
                return Dual(self.re, self.im)
            else:
                raise KeyError

    def __setitem__(self, key, value):
        value = to_dual(value)
        try:
            self.re[key] = value.re
            self.im[key] = value.im
        except:
            if self.__len__() == 1:
                self.re = value.re
                self.im = value.im
            else:
                raise KeyError

    def __delitem__(self, key):
        try:
            self.re[key] = 0
            self.im[key] = 0
        except:
            if self.__len__() == 1:
                self.re = 0
                self.im = 0
            else:
                raise KeyError

    def __iter__(self):
        self._i = 0
        while self._i < self.__len__():
            yield Dual(self.re[self._i], self.im[self._i])
            self._i += 1

    def __next__(self):
        if self._i > self.__len__():
            raise StopIteration
        self._i += 1
        return Dual(self.re[self._i - 1], self.im[self._i - 1])

    def __reversed__(self):
        self._i = self.__len__() - 1
        while self._i >= 0:
            yield Dual(self.re[self._i], self.im[self._i])
            self._i -= 1

    def __contains__(self, item):
        return np.any(self.__eq__(item))

    def __add__(self, other):
        other = to_dual(other)
        return Dual(self.re + other.re, self.im + other.im)

    def __sub__(self, other):
        other = to_dual(other)
        return Dual(self.re - other.re, self.im - other.im)

    def __mul__(self, other):
        other = to_dual(other)
        return Dual(self.re * other.re, self.im * other.re + self.re * other.im)

    def __matmul__(self, other):
        """
        python3.5以上かつNumpy1.10以上で実装されている行列積演算子@の動作定義
        Define behavior of matrix product operator '@' implemented in python3.5 or higher and Numpy 1.10 or higher.
        """
        other = to_dual(other)

        # 全て行列に変換する
        # Convert all to matrix.
        if np.ndim(self.re) == 0:
            # 点 point
            self_shape = "point"
            self_buf = Dual(np.array([[self.re]]), np.array([[self.im]]))
        elif np.ndim(self.re) == 1:
            # ベクトル vector
            self_shape = "vector"
            self_buf = Dual(np.array([self.re]), np.array([self.im]))
        else:
            # 行列 matrix
            self_shape = "matrix"
            self_buf = Dual(np.copy(self.re), np.copy(self.im))
        if np.ndim(other.re) == 0:
            other_shape = "point"
            other_buf = Dual(np.array([[other.re]]), np.array([[other.im]]))
        elif np.ndim(other.re) == 1:
            other_shape = "vector"
            other_buf = Dual(np.array([other.re]), np.array([other.im]))
        else:
            other_shape = "matrix"
            other_buf = Dual(np.copy(other.re), np.copy(other.im))

        # 行列積 matrix product
        # 行列積の制約を満たさない計算はエラーが起こる。
        # Calculations that do not satisfy the matrix product constraint will result in an error.
        cal_buf = Dual(self_buf.re @ other_buf.re, \
                       self_buf.im @ other_buf.re + self_buf.re @ other_buf.im)

        # 形状復元 Shape restoration
        # (m, n) <= m行n列 'm':number of row, 'n':number of column
        if self_shape == "point":
            if other_shape == "point":
                # (1, 1) @ (1, 1) = (1, 1) => 点 point
                return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
            elif other_shape == "vector":
                # (1, 1) @ (1, n) = (1, n) => 点       point  (n == 1)
                #                             ベクトル vector (n != 1)
                if other_buf.re.shape[1] == 1:
                    return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
                else:
                    return Dual(cal_buf.re[0], cal_buf.im[0])
            else:
                # (1, 1) @ (1, n) = (1, n) => 点       point  (n == 1)
                #                             ベクトル vector (n != 1)
                if other_buf.re.shape[1] == 1:
                    return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
                else:
                    return Dual(cal_buf.re[0], cal_buf.im[0])
        elif self_shape == "vector":
            # (1, m) @ (m, n) = (1, n) => 点       point  (n == 1)
            #                             ベクトル vector (n != 1)
            if other_buf.re.shape[1] == 1:
                return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
            else:
                return Dual(cal_buf.re[0], cal_buf.im[0])
        else:
            # (l, m) @ (m, n) = (l, m) => 点       point   (l == 1 and n == 1)
            #                             ベクトル vecgtor (l == 1 and n != 1)
            #                             行列     matrix  (l != 1)
            if self_buf.re.shape[0] == 1:
                if other_buf.re.shape[1] == 1:
                    return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
                else:
                    return Dual(cal_buf.re[0], cal_buf.im[0])
            else:
                return cal_buf

    def __truediv__(self, other):
        other = to_dual(other)
        d = other.re * other.re
        if np.all(~d):
            raise ZeroDivisionError("math domain error")
        return Dual(self.re * other.re / d, (self.im * other.re - self.re * other.im) / d)

    def __floordiv__(self, other):
        raise TypeError("can't take floor of dual number.")

    def __mod__(self, other):
        raise TypeError("can't take mod of dual number.")

    def __divmod__(self, other):
        raise TypeError("can't take floor and mod of dual number.")

    def __pow__(self, other):
        if is_dual(other):
            if other.im:
                if self.im:
                    raise TypeError("Dual to the Dual power.")
                else:
                    return Dual(1, other.im * np.log(self.re))
            other = other.re
        return Dual(np.power(self.re, other), other * np.power(self.re, other - 1) * self.im)

    def __lshift__(self, other):
        """
        シフト演算はサポートされません。
        Be unsupported shift operator.
        """
        raise NotImplemented

    def __rshift__(self, other):
        """
        シフト演算はサポートされません。
        Be unsupported shift operator.
        """
        raise NotImplemented

    def __and__(self, other):
        """
        論理演算はサポートされません。
        Be unsupported logical operator.
        """
        raise NotImplemented

    def __xor__(self, other):
        """
        論理演算はサポートされません。
        Be unsupported logical operator.
        """
        raise NotImplemented

    def __or__(self, other):
        """
        論理演算はサポートされません。
        Be unsupported logical operator.
        """
        raise NotImplemented

    __radd__ = __add__

    def __rsub__(self, other):
        other = to_dual(other)
        return other - self

    __rmul__ = __mul__

    def __rtruediv__(self, other):
        other = to_dual(other)
        return other / self

    __rfloordiv__ = __floordiv__
    __rmod__ = __mod__
    __rdivmod__ = __divmod__

    def __rpow__(self, other):
        other = to_dual(other)
        return other ** self

    __rlshift__ = __lshift__
    __rrshift__ = __rshift__
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    def __iadd__(self, other):
        self.assign(self + other)
        return self

    def __isub__(self, other):
        self.assign(self - other)
        return self

    def __imul__(self, other):
        self.assign(self * other)
        return self

    def __imatmul__(self, other):
        self.assign(self @ other)
        return self

    def __itruediv__(self, other):
        self.assign(self / other)
        return self

    __ifloordiv__ = __floordiv__
    __imod__ = __mod__

    def __ipow__(self, other):
        self.assign(self ** other)
        return self

    __ilshift__ = __lshift__
    __irshift__ = __rshift__
    __iand__ = __and__
    __ixor__ = __xor__
    __ior__ = __or__

    def __neg__(self):
        return Dual(- self.re, - self.im)

    def __pos__(self):
        return self

    def __abs__(self):
        return np.hypot(self.re, self.im)

    def __invert__(self):
        """
        ビット反転はサポートされません。
        Be unsupported bit reverse.
        """
        raise NotImplemented

    def __complex__(self):
        return np.complex(self.re, self.im)

    def __int__(self):
        if self.im:
            return Dual(np.int(self.re), np.int(self.im))
        return np.int(self.re)

    def __float__(self):
        if self.im:
            return Dual(np.float(self.re), np.float(self.im))
        return np.float(self.re)

    def __round__(self, ndigits=0):
        return Dual(np.round(self.re, decimal=ndigits), np.round(self.im, decimal=ndigits))

    def __trunc__(self):
        return Dual(np.trunc(self.re), np.trunc(self.im))

    def __floor__(self):
        return Dual(np.floor(self.re), np.floor(self.im))

    def __ceil__(self):
        return Dual(np.ceil(self.re), np.ceil(self.im))

    def __nonzero__(self):
        return ~ ((self.re == 0) & (self.im == 0))

    def assign(self, other):
        other = to_dual(other)
        self.re = other.re
        self.im = other.im

    @property
    def T(self):
        try:
            if np.ndim(self.re) == 1:
                return Dual(np.array([self.re]).T, np.array([self.im]).T)
            else:
                return Dual(self.re.T, self.im.T)
        except:
            return Dual(self.re, self.im)


def is_dual(obj):
    return hasattr(obj, "re") and hasattr(obj, "im")

def to_dual(obj):
    if is_dual(obj):
        return obj
    elif isinstance(obj, tuple):
        return Dual(*obj)
    else:
        return Dual(obj)

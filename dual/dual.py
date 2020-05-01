#
# @Author: Yuma Shimomoto (2020)
# @email: yuma_1215@docomo.ne.jp
#
"""
参考：https://github.com/tmurakami1234/my_python_module/blob/master/dual/dual.py
      https://www.pythonprogramming.in/example-of-reversed-magic-method-in-python.html
"""

import numpy as np


class dual():
    def __init__(self, re=0, im=0):
        if not isinstance(re, np.ndarray):
            re = np.array(re)
        if not isinstance(im, np.ndarray):
            im = np.array(im)
        im = np.broadcast_to(re.shape)

        _re = np.zeros(re.shape)
        _im = np.zeres(im.shape)

        if is_dual(re):
            _re = re.re
            _im = re.im
        else:
            _re = re
        if is_dual(im):
            _re -= im.im
            _im += im.re
        else:
            _im += im

        self.__dict__["re"] = _re
        self.__dict__["im"] = _im

    def __repr__(self):
        if not self.im:
            return 'dual({})'.format(self.re)
        else:
            return 'dual({},{})'.format(self.re, self.im)

    def __str__(self):
        if not self.im:
            return repr(self.re)
        else:
            return 'dual({},{})'.format(self.re, self.im)

    def __lt__(self, other):
        """
        「実部を比較=>虚部を比較」(辞書順)で比較しています。
        ノルム順にする場合はコメントアウトしているものを使ってください。
        Compared by 'compare real => compare imaginary' (dictionary-order).
        If use norm-order, activate commentouted one.
        """
        other = to_dual(other)
        return (self.re < other.re) or ((self.re == other.re) and self.im < other.im)

#    def __lt__(self, other):
#        """
#        ノルム順で比較しています。
#        Compared by norm-order.
#        """
#        other = to_dual(other)
#        return self.__abs__() < other.__abs__()

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __eq__(self, other):
        other = to_dual(other)
        return self.re == other.re and self.im == other.im

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __hash__(self):
        if not self.im:
            return hash(self.re)
        return hash((self.re, self.im))

    def __bool__(self):
        return self.__nonzero__()

    def __setattr__(self, name, value):
        raise TypeError('Dual numbers are immutable.')

    def __len__(self):
        return len(self.re)

    def __getitem__(self, key):
        return Dual(self.re[key], self.im[key])

    def __setitem__(self, key, value):
        value = to_dual(value)
        self.re[key] = value.re
        self.im[key] = value.im

    def __delitem__(self, key):
        self.re[key] = 0
        self.im[key] = 0

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
            yield Dual(self.re[self._i]. self.im[self._i])
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
        Define behavior of matrix multiplication operator '@' implemented in python3.5 or higher and Numpy 1.10 or higher.
        """
        other = to_dual(other)
        return Dual(self.re @ other.re, self.im @ other.re + self.re @ other.im)

    def __truediv__(self, other):
        other = to_dual(other)
        d = other.re * other.re
        if not d:
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

    def assign(self, other):
        other = to_dual(other)
        self.re = other.re
        self.im = other.im

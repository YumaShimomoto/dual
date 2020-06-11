#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#
"""
参考：https://github.com/tmurakami1234/my_python_module/blob/master/dual/dual.py
      https://www.pythonprogramming.in/example-of-reversed-magic-method-in-python.html
"""
import numpy as np
import math


class Dual():
    def __init__(self, re=0, im=0, dtype=np.float):
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
                # Argument 're' is "Type isn't 'Dual'"
                # and "Type is 'ndarray'"
                re_re = re
                re_im = np.zeros(re.shape)
            else:
                re_re = np.array(re)
                re_im = np.zeros(re_re.shape)
        if is_dual(im):
            if isinstance(im.re, np.ndarray):
                # 引数imが「Dual型である」かつ「ndarray型である」
                # Argument 'im' is "Type is 'Dual'"
                # and "Type is 'ndarray'"
                im_re = im.re
                im_im = im.im
            else:
                im_re = np.array(im.re)
                im_im = np.array(im.im)
        else:
            if isinstance(im, np.ndarray):
                # 引数imが「Dual型ではない」かつ「ndarray型である」
                # Argument 'im' is "Type isn't 'Dual'"
                # and "Type is 'ndarray'"
                im_re = im
                im_im = np.zeros(im.shape)
            else:
                im_re = np.array(im)
                im_im = np.zeros(im_re.shape)

        # re = x + ex', im = y + ey' の時   # When "re = ~~, im = ~~"
        # Dual(re, im) = re + eim
        #              = (x + ex') + e(y + ey')
        #              = x + e(x' + y')
        _re = np.array(re_re, dtype=dtype)
        _im = np.array(re_im + im_re, dtype=dtype)

        self.__dict__["re"] = _re
        self.__dict__["im"] = _im
        self.__dict__["_i"] = 0
        self.__dict__["_re"] = _re.reshape(-1)
        self.__dict__["_im"] = _im.reshape(-1)


    def __repr__(self):
        return str(self.complex).replace("j", "e")


    def __str__(self):
        return str(self.complex).replace("j", "e")


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
        return (self < other) | (self == other)


    def __eq__(self, other):
        other = to_dual(other)
        return (self.re == other.re) & (self.im == other.im)


    def __ne__(self, other):
        return ~ (self == other)


    def __gt__(self, other):
        return ~ (self <= other)


    def __ge__(self, other):
        return ~ (self < other)


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
            if len(self) == 1:
                return Dual(self.re, self.im)
            else:
                raise KeyError


    def __setitem__(self, key, value):
        value = to_dual(value)
        try:
            self.re[key] = value.re
            self.im[key] = value.im
        except:
            if len(self) == 1:
                self.re = value.re
                self.im = value.im
            else:
                raise KeyError


    def __delitem__(self, key):
        try:
            self.re[key] = 0
            self.im[key] = 0
        except:
            if len(self) == 1:
                self.re = 0
                self.im = 0
            else:
                raise KeyError


    def __iter__(self):
        self._i = 0
        return self


    def __next__(self):
        if self._i >= len(self._re):
            raise StopIteration
        self._i += 1
        return Dual(self._re[self._i - 1], self._im[self._i - 1])


    def __reversed__(self):
        self._i = 0
        return Dual(self._re[::-1], self._im[::-1])


    def __contains__(self, item):
        return np.any(self == item)


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
        python3.5以上かつNumpy1.10以上で実装されている
        行列積演算子@の動作定義
        Define behavior of matrix product operator '@'
        implemented in python3.5 or higher and Numpy 1.10 or higher.
        """
        other = to_dual(other)

        # 全て行列に変換する
        # Convert all to matrix.
        if self.ndim == 0:
            # 点 point
            self_shape = "point"
            self_buf = Dual(np.array([[self.re]]), np.array([[self.im]]))
        elif self.ndim == 1:
            # ベクトル vector
            self_shape = "vector"
            self_buf = Dual(np.array([self.re]), np.array([self.im]))
        else:
            # 行列 matrix
            self_shape = "matrix"
            self_buf = Dual(np.copy(self.re), np.copy(self.im))
        if other.ndim == 0:
            other_shape = "point"
            other_buf = Dual(np.array([[other.re]]),
                             np.array([[other.im]]))
        elif other.ndim == 1:
            other_shape = "vector"
            other_buf = Dual(np.array([other.re]), np.array([other.im]))
        else:
            other_shape = "matrix"
            other_buf = Dual(np.copy(other.re), np.copy(other.im))

        # 行列積 matrix product
        # 行列積の制約を満たさない計算はエラーが起こる。
        # Calculations that do not satisfy the matrix product constraint
        # will result in an error.
        cal_buf = Dual(self_buf.re @ other_buf.re,
                       self_buf.im @ other_buf.re
                       + self_buf.re @ other_buf.im)

        # 形状復元 Shape restoration
        # (m, n) <= m行n列 'm':number of row, 'n':number of column
        if self_shape == "point":
            if other_shape == "point":
                # (1, 1) @ (1, 1) = (1, 1) => 点 point
                return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
            elif other_shape == "vector":
                # (1, 1) @ (1, n) = (1, n) => 点       point  (n == 1)
                #                             ベクトル vector (n != 1)
                if other_buf.shape[1] == 1:
                    return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
                else:
                    return Dual(cal_buf.re[0], cal_buf.im[0])
            else:
                # (1, 1) @ (1, n) = (1, n) => 点       point  (n == 1)
                #                             ベクトル vector (n != 1)
                if other_buf.shape[1] == 1:
                    return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
                else:
                    return Dual(cal_buf.re[0], cal_buf.im[0])
        elif self_shape == "vector":
            # (1, m) @ (m, n) = (1, n) => 点       point  (n == 1)
            #                             ベクトル vector (n != 1)
            if other_buf.shape[1] == 1:
                return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
            else:
                return Dual(cal_buf.re[0], cal_buf.im[0])
        else:
            # (l, m) @ (m, n) = (l, m) => 点       point   (l == 1 and n == 1)
            #                             ベクトル vecgtor (l == 1 and n != 1)
            #                             行列     matrix  (l != 1)
            if self_buf.shape[0] == 1:
                if other_buf.shape[1] == 1:
                    return Dual(cal_buf.re[0][0], cal_buf.im[0][0])
                else:
                    return Dual(cal_buf.re[0], cal_buf.im[0])
            else:
                return cal_buf


    def __truediv__(self, other):
        other = to_dual(other)
        d = other.re * other.re
        if not np.all(d):
            raise ZeroDivisionError("math domain error")
        return Dual(self.re * other.re / d,
                    (self.im * other.re - self.re * other.im) / d)


    def __floordiv__(self, other):
        raise TypeError("can't take floor of dual number.")


    def __mod__(self, other):
        raise TypeError("can't take mod of dual number.")


    def __divmod__(self, other):
        raise TypeError("can't take floor and mod of dual number.")


    def __pow__(self, other):
        """
        マクローリン展開すると Do McLaughlin expansion
            exp(e x) = 1 + e x
        以降の式変形の準備 and ready for following transration;
            Z = z + e z' = exp(log(Z))
            log(Z) = log(|Z| exp(e arg(Z))) = log|Z| + e arg(Z)
            arg(Z) = arctan(z' / z)
        以下を仮定 supposed;
            X = x + e x'
            Y = y + e y'

        X ** Y = exp(log(X)) ** Y
               = exp(Y log(X))
               = exp((y + e y')(log|X| + e arg(X)))
               = exp(y log|X| + e (y' log|X| + y arg(X)))
               = (|X| ** y)(1 + e (y' log|X| + y arg(X)))
               = a + e a b
        a = |X| ** y
        b = y' log|X| + y arg(X)

        if x' = 0 and y' = 0 then
            a = |X| ** y = |x| ** y => x ** y
            arg(X) = 0
            b = 0
            X ** Y = a
        else:
            上記の通りa, b, arg(X)それぞれを計算する。
            As mentioned above after calculating each buf value;
                a, b, arg(X).
        """
        if is_dual(other):
            # otherがDual型の場合
            # if the type of 'other' is 'Dual',
            if self.ndim == 0 and other.ndim == 0:
                # 点同士の演算はここで行う
                # Do point-to-point calculations here.
                # ndarrayじゃなければNumpyよりもmathの方が早い
                # If it is not ndarray, 'math' is faster than 'Numpy'.
                if other.im:
                    if self.im:
                        alpha = abs(self) ** other.re
                        argX = math.atan(self.im / self.re)
                        beta = (other.im * math.log(abs(self))
                                + other.re * argX)
                    else:
                        alpha = self.re ** other.re
                        beta = other.im * math.log(abs(self.re))
                    return Dual(alpha, alpha * beta)
                else:
                    if self.im:
                        alpha = abs(self) ** other.re
                        argX = math.arctan(self.im / self.re)
                        beta = other.re * argX
                        return Dual(alpha, alpha * beta)
                    else:
                        # other.imもself.imも0の場合は
                        # 数値に変換して計算させる。
                        # If 'other.im' and 'self.im' is 0,
                        # calculate it by converting it
                        # to a numerical value.
                        other = other.re
            else:
                # それ以外の場合は定義通り計算する。
                # Otherwise compute as defined.
                absX = abs(self)
                alpha = absX ** other.re
                argX = np.arctan(self.im / self.re)
                beta = other.im * np.log(absX) + other.re * argX
                return Dual(alpha, alpha * beta)
        # otherがDual型ではない場合は普通に計算して返す。
        # If the type of 'other' isn't 'Dual', return Numpy calculations.
        return Dual(np.power(self.re, other),
                    other * np.power(self.re, other - 1) * self.im)


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
        if self.ndim != 0:
            raise ValueError(
                    "Can't convert Dual with vector or matrix to complex.")
        return complex(self.re, self.im)


    def __int__(self):
        if np.any(self.im):
            raise ValueError("Can't convert Dual with nonzero im to int.")
        return int(self.re)


    def __float__(self):
        if np.any(self.im):
            raise ValueError(
                    "Can't convert Dual with nonzero im to float.")
        return float(self.re)


    def __round__(self, ndigits=0):
        return Dual(np.round(self.re, decimal=ndigits),
                    np.round(self.im, decimal=ndigits))


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


    def broadcast_to(self, shape):
        return Dual(np.broadcast_to(self.re, shape),
                    np.broadcast_to(self.im, shape))


    def reshape(self, shape, order="C"):
        return Dual(np.reshape(self.re, shape, order=order), \
                    np.reshape(self.im, shape, order=order))


    def squeeze(self, axis=None):
        return Dual(np.squeeze(self.re, axis=axis),
                    np.squeeze(self.im, axis=axis))


    def ravel(self, order="C"):
        return Dual(np.ravel(self.re, order=order),
                    np.ravel(self.im, order=order))


    def flatten(self, order="C"):
        try:
            return Dual(self.re.flatten(order=order),
                        self.im.flatten(order=order))
        except:
            return self


    def transpose(self, axes=None):
        return Dual(np.transpose(self.re, axes=axes),
                    np.transpose(self.im, axes=axes))


    def diagonal(self, offset=0, axis1=0, axis2=1):
        return Dual(self.re.diagonal(offset=offset,
                                     axis1=axis1, axis2=axis2),
                    self.im.diagonal(offset=offset,
                                     axis1=axis1, axis2=axis2))


    def astype(self, dtype, order='K', casting='unsafe',
               subok=True, copy=True):
        return Dual(self.re.astype(dtype, order=order, casting=casting,
                                   subok=subok, copy=copy),
                    self.im.astype(dtype, order=order, casting=casting,
                                   subok=subok, copy=copy))


    @property
    def complex(self):
        try:
            return np.array(
                    [complex(x.re, x.im) for x in self]).reshape(self.shape)
        except:
            return complex(self.re, self.im)


    @property
    def int(self):
        if np.any(self.im):
            return Dual(self.re, self.im, dtype=np.int)
        return np.array(self.re, dtype=np.int)


    @property
    def float(self):
        if np.any(self.im):
            return Dual(self.re, self.im, dtype=np.float)
        return np.array(self.re, dtype=np.float)


    @property
    def T(self):
        try:
            if np.ndim(self.re) == 1:
                return Dual(np.array([self.re]).T, np.array([self.im]).T)
            else:
                return Dual(self.re.T, self.im.T)
        except:
            return Dual(self.re, self.im)


    @property
    def ndim(self):
        return np.ndim(self.re)


    @property
    def size(self):
        return np.size(self.re)


    @property
    def shape(self):
        return self.re.shape


def is_dual(obj):
    return hasattr(obj, "re") and hasattr(obj, "im")


def to_dual(obj):
    if is_dual(obj):
        return obj
    elif isinstance(obj, tuple):
        return Dual(*obj)
    else:
        return Dual(obj)

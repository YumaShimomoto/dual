#
# @Author: kuroitu (2020)
# @email: skuroitu@gmail.com
#

import matplotlib.pyplot as plt
import numpy as np
import time


import dual


def test():
    print("+------------------------------+")
    print("| Test of dmath module by plot |")
    print("+------------------------------+\n")

    delta = 5e-2
    x = dual.Dual(np.arange(-3, 3, delta), 1)
    x2 = dual.Dual(np.arange(-1, 5, delta), 1)

    def set_subplot(ax, title, x, y, delta=5e-2):
        ax.set_title(title)
        re_Q1, re_Q3 = np.percentile(y.re, [25, 75])
        re_IQR = re_Q3 - re_Q1
        re_lower = re_Q1 - 1.5 * re_IQR
        re_higher = re_Q3 + 1.5 * re_IQR
        im_Q1, im_Q3 = np.percentile(y.im, [25, 75])
        im_IQR = im_Q3 - im_Q1
        im_lower = im_Q1 - 1.5 * im_IQR
        im_higher = im_Q3 + 1.5 * im_IQR
        lower_border = min(max(re_lower, np.nanmin(y.re)), max(im_lower, np.nanmin(y.im)))
        higher_border = max(min(re_higher, np.nanmax(y.re)), min(im_higher, np.nanmax(y.im)))
        y.re = np.where(y.re < lower_border, np.nan, np.where(y.re > higher_border, np.nan, y.re))
        y.im = np.where(y.im < lower_border, np.nan, np.where(y.im > higher_border, np.nan, y.im))
        for i in range(1, len(x)):
            if x.re[i] - x.re[i - 1] > delta:
                y.re[i]= np.nan
                y.im[i] = np.nan
            elif y.re[i] * y.re[i - 1] < -1:
                y.re[i] = np.nan
                if y.im[i] * y.im[i - 1] < -1:
                    y.im[i] = np.nan
            elif y.im[i] * y.im[i - 1] < -1:
                y.im[i] = np.nan
                if y.re[i] * y.re[i - 1] < -1:
                    y.re[i] = np.nan

        ax.plot(x.re, y.re)
        ax.plot(x.re, y.im)
        ax.set_ylim(lower_border, higher_border)
        ax.grid()

    def compare_time(title, x, func, n, **kwds):
        print(title)
        if len(kwds):
            start1 = time.time()
            for i in range(n):
                y = func(x, **kwds)
            end1 = time.time()
            start2 = time.time()
            for i in range(n):
                y = func(x, s=False, **kwds)
            end2 = time.time()
            print("\t", end1 - start1, end2 - start2)
            if end1 - start1 <= end2 - start2:
                print("\t", "True faster than False")
            else:
                print("\t", "False faster than True")
            print("\t re:", np.all(np.isclose(func(x, **kwds).re, func(x, s=False, **kwds).re)))
            print("\t im:", np.all(np.isclose(func(x, **kwds).im, func(x, s=False, **kwds).im)))
            #print(func(x, *args) - func(x, *args, False))
        else:
            start1 = time.time()
            for i in range(n):
                y = func(x)
            end1 = time.time()
            start2 = time.time()
            for i in range(n):
                y = func(x, s=False)
            end2 = time.time()
            print("\t", end1 - start1, end2 - start2)
            if end1 - start1 <= end2 - start2:
                print("\t", "True faster than False")
            else:
                print("\t", "False faster than True")
            print("\t re:", np.all(np.isclose(func(x).re, func(x, s=False).re)))
            print("\t im:", np.all(np.isclose(func(x).im, func(x, s=False).im)))
            #print(func(x) - func(x, False))

    print("exponential.pyのテスト", "test exponential.py", "\n")
    fig, ax = plt.subplots(3, 3)
# https://www.wolframalpha.com/input/?i=x%5E3%2C+3x%5E2%2C+x+%3D+-3+to+3&lang=ja
    y = dual.power(x, 3)
    set_subplot(ax[0, 0], "power(x, 3)", x, y)
# https://www.wolframalpha.com/input/?i=x%5E2%2C+2x%2C+x+%3D+-3+to+3&lang=ja
    y = dual.square(x)
    set_subplot(ax[0, 1], "square(x)", x, y)
# https://www.wolframalpha.com/input/?i=sqrt%28x%29%2C+d%2Fdx+sqrt%28x%29%2C+x+%3D+1+to+3&lang=ja
    y = dual.sqrt(x[x >= 1])
    set_subplot(ax[0, 2], "sqrt(x)", x[x >= 1], y)
# https://www.wolframalpha.com/input/?i=cbrt%28x%29%2C+d%2Fdx+cbrt%28x%29%2C+x+%3D+1+to+3&lang=ja
    y = dual.cbrt(x[x >= 1])
    set_subplot(ax[1, 0], "cbrt(x)", x[x >= 1], y)
# https://www.wolframalpha.com/input/?i=exp%28x%29%2C+d%2Fdx+exp%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.exp(x)
    set_subplot(ax[1, 1], "exp(x)", x, y)
# https://www.wolframalpha.com/input/?i=2%5Ex%2C+d%2Fdx+2%5Ex%2C+x+%3D+-3+to+3&lang=ja
    y = dual.exp2(x)
    set_subplot(ax[1, 2], "exp2(x)", x, y)
# https://www.wolframalpha.com/input/?i=exp%28x%29-1%2C+d%2Fdx+exp%28x%29-1%2C+x+%3D+-3+to+3&lang=ja
    y = dual.expm1(x)
    set_subplot(ax[2, 0], "expm1(x)", x, y)
    fig.tight_layout()
    plt.show()
    fig.savefig("exponential.png")

    print("logarithm.pyのテスト", "test logarithm.py", "\n")
    fig, ax = plt.subplots(3, 3)
# https://www.wolframalpha.com/input/?i=log%28x%29%2C+d%2Fdx+log%28x%29%2C+x+%3D+0.05+to+3&lang=ja
    y = dual.log(x[x > 0])
    set_subplot(ax[0, 0], "log(x)", x[x > 0], y)
# https://www.wolframalpha.com/input/?i=log_3%28x%29%2C+d%2Fdx+log_3%28x%29%2C+x+%3D+0.05+to+3&lang=ja
    y = dual.logn(x[x > 0], 3)
    set_subplot(ax[0, 1], "logn(x, 3)", x[x > 0], y)
# https://www.wolframalpha.com/input/?i=log_7%28x%29%2C+d%2Fdx+log_7%28x%29%2C+x+%3D+0.05+to+3&lang=ja
    y = dual.logn(x[x > 0], 7)
    set_subplot(ax[0, 2], "logn(x, 7)", x[x > 0], y)
# https://www.wolframalpha.com/input/?i=log_2%28x%29%2C+d%2Fdx+log_2%28x%29%2C+x+%3D+0.05+to+3&lang=ja
    y = dual.log2(x[x > 0])
    set_subplot(ax[1, 0], "log2(x)", x[x > 0], y)
# https://www.wolframalpha.com/input/?i=log_10%28x%29%2C+d%2Fdx+log_10%28x%29%2C+x+%3D+0.05+to+3&lang=ja
    y = dual.log10(x[x > 0])
    set_subplot(ax[1, 1], "log10(x)", x[x > 0], y)
# https://www.wolframalpha.com/input/?i=log%28x%2B1%29%2C+d%2Fdx+log%28x%2B1%29%2C+x+%3D+-0.95+to+3&lang=ja
    y = dual.log1p(x[x > -1])
    set_subplot(ax[1, 2], "log1p(x)", x[x > -1], y)
# https://www.wolframalpha.com/input/?i=log%28exp%28x%29+%2B+exp%28x+%2B+2%29%29%2C+d%2Fdx+log%28exp%28x%29+%2B+exp%28x+%2B+2%29%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.logaddexp(x, x2)
    set_subplot(ax[2, 0], "logaddexp(x, x2)", x, y)
# https://www.wolframalpha.com/input/?i=log%282%5E%28x%29+%2B+2%5E%28x+%2B+2%29%29%2C+d%2Fdx+log%282%5E%28x%29+%2B+2%5E%28x+%2B+2%29%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.logaddexp2(x, x2)
    set_subplot(ax[2, 1], "logaddexp2(x, x2)", x, y)
    fig.tight_layout()
    plt.show()
    fig.savefig("logarithm.png")

    print("trigonometric.pyのテスト","test trigonometric.py","\n")
    fig, ax = plt.subplots(4, 4)
# https://www.wolframalpha.com/input/?i=sin%28x%29%2C+d%2Fdx+sin%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sin(x)
    set_subplot(ax[0, 0], "sin(x)", x, y)
# https://www.wolframalpha.com/input/?i=cos%28x%29%2C+d%2Fdx+cos%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.cos(x)
    set_subplot(ax[0, 1], "cos(x)", x, y)
# https://www.wolframalpha.com/input/?i=tan%28x%29%2C+d%2Fdx+tan%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.tan(x)
    set_subplot(ax[0, 2], "tan(x)", x, y)
# https://www.wolframalpha.com/input/?i=arcsin%28x%29%2C+d%2Fdx+arcsin%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arcsin(x[(-1 < x) * (x < 1)])
    set_subplot(ax[0, 3], "arcsin(x)", x[(-1 < x) * (x < 1)], y)
# https://www.wolframalpha.com/input/?i=arccos%28x%29%2C+d%2Fdx+arccos%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arccos(x[(-1 < x) * (x < 1)])
    set_subplot(ax[1, 0], "arccos(x)", x[(-1 < x) * (x < 1)], y)
# https://www.wolframalpha.com/input/?i=arctan%28x%29%2C+d%2Fdx+arctan%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arctan(x)
    set_subplot(ax[1, 1], "arctan(x)", x, y)
# https://www.wolframalpha.com/input/?i=arctan%28x+%2F+%28x+%2B+2%29%29%2C+d%2Fdx+arctan%28x+%2F+%28x+%2B+2%29%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arctan2(x, x2)
    set_subplot(ax[1, 2], "arctan2(x, x2)", x, y)
# https://www.wolframalpha.com/input/?i=csc%28x%29%2C+d%2Fdx+csc%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.csc(x)
    set_subplot(ax[1, 3], "csc(x)", x, y)
# https://www.wolframalpha.com/input/?i=sec%28x%29%2C+d%2Fdx+sec%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sec(x)
    set_subplot(ax[2, 0], "sec(x)", x, y)
# https://www.wolframalpha.com/input/?i=cot%28x%29%2C+d%2Fdx+cot%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.cot(x)
    set_subplot(ax[2, 1], "cot(x)", x, y)
# https://www.wolframalpha.com/input/?i=arccsc%28x%29%2C+d%2Fdx+arccsc%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arccsc(x[(x <= -1) + (1 <= x)])
    set_subplot(ax[2, 2], "arccsc(x)", x[(x <= -1) + (1 <= x)], y)
# https://www.wolframalpha.com/input/?i=arcsec%28x%29%2C+d%2Fdx+arcsec%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arcsec(x[(x <= -1) + (1 <= x)])
    set_subplot(ax[2, 3], "arcsec(x)", x[(x <= -1) + (1 <= x)], y)
# https://www.wolframalpha.com/input/?i=arccot%28x%29%2C+d%2Fdx+arccot%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arccot(x)
    set_subplot(ax[3, 0], "arccot(x)", x, y)
# https://www.wolframalpha.com/input/?i=sinc%28x+*+pi%29%2C+d%2Fdx+sinc%28x+*+pi%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sinc(x)
    set_subplot(ax[3, 1], "sinc(x)", x, y)
# https://www.wolframalpha.com/input/?i=sinc%28x%29%2C+d%2Fdx+sinc%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sinc(x, norm=False)
    set_subplot(ax[3, 2], "sinc(x, False)", x, y)
    fig.tight_layout()
    plt.show()
    fig.savefig("trigonometric.png")

    print("hyperbolic.pyのテスト","test hyperbolic.py","\n")
    fig, ax = plt.subplots(4, 4)
# https://www.wolframalpha.com/input/?i=sinh%28x%29%2C+d%2Fdx+sinh%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sinh(x)
    set_subplot(ax[0, 0], "sinh(x)", x, y)
# https://www.wolframalpha.com/input/?i=cosh%28x%29%2C+d%2Fdx+cosh%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.cosh(x)
    set_subplot(ax[0, 1], "cosh(x)", x, y)
# https://www.wolframalpha.com/input/?i=tanh%28x%29%2C+d%2Fdx+tanh%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.tanh(x)
    set_subplot(ax[0, 2], "tanh(x)", x, y)
# https://www.wolframalpha.com/input/?i=arcsinh%28x%29%2C+d%2Fdx+arcsinh%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arcsinh(x)
    set_subplot(ax[0, 3], "arcsinh(x)", x, y)
# https://www.wolframalpha.com/input/?i=arccosh%28x%29%2C+d%2Fdx+arccosh%28x%29%2C+x+%3D+1+to+3&lang=ja
    y = dual.arccosh(x[x >= 1])
    set_subplot(ax[1, 0], "arccosh(x)", x[x >= 1], y)
# https://www.wolframalpha.com/input/?i=arctanh%28x%29%2C+d%2Fdx+arctanh%28x%29%2C+x+%3D+-1+to+1&lang=ja
    y = dual.arctanh(x[(-1 < x) * (x < 1)])
    set_subplot(ax[1, 1], "arctanh(x)", x[(-1 < x) * (x < 1)], y)
# https://www.wolframalpha.com/input/?i=csch%28x%29%2C+d%2Fdx+csch%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.csch(x[x != 0])
    set_subplot(ax[1, 2], "csch(x)", x[x != 0], y)
# https://www.wolframalpha.com/input/?i=sech%28x%29%2C+d%2Fdx+sech%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sech(x)
    set_subplot(ax[1, 3], "sech(x)", x, y)
# https://www.wolframalpha.com/input/?i=coth%28x%29%2C+d%2Fdx+coth%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.coth(x[x != 0])
    set_subplot(ax[2, 0], "coth(x)", x[x != 0], y)
# https://www.wolframalpha.com/input/?i=arccsch%28x%29%2C+d%2Fdx+arccsch%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arccsch(x[x != 0])
    set_subplot(ax[2, 1], "arccsch(x)", x[x != 0], y)
# https://www.wolframalpha.com/input/?i=arcsech%28x%29%2C+d%2Fdx+arcsech%28x%29%2C+x+%3D+0+to+1&lang=ja
    y = dual.arcsech(x[(0 < x) * (x <= 1)])
    set_subplot(ax[2, 2], "arcsech(x)", x[(0 < x) * (x <= 1)], y)
# https://www.wolframalpha.com/input/?i=arccoth%28x%29%2C+d%2Fdx+arccoth%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.arccoth(x[(x < -1) + (1 < x)])
    set_subplot(ax[2, 3], "arccoth(x)", x[(x < -1) + (1 < x)], y)
# https://www.wolframalpha.com/input/?i=sinh%28x+*+pi%29+%2F+%28x+*+pi%29%2C+d%2Fdx+sinh%28x+*+pi%29+%2F+%28x+*+pi%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sinch(x[x != 0])
    set_subplot(ax[3, 0], "sinch(x)", x[x != 0], y)
# https://www.wolframalpha.com/input/?i=sinh%28x%29+%2F+%28x%29%2C+d%2Fdx+sinh%28x%29+%2F+%28x%29%2C+x+%3D+-3+to+3&lang=ja
    y = dual.sinch(x[x != 0], norm=False)
    set_subplot(ax[3, 1], "sinch(x, False)", x[x != 0], y)
    fig.tight_layout()
    plt.show()
    fig.savefig("hyperbolic.png")


if __name__ == "__main__":
    test()

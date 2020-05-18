import os
import numpy as np


from ..dual import *


##########
# 配列ユーティリティ関数 array-utility function
#######
def ndim(obj):
    return np.ndim(obj.re)


def shape(obj):
    return np.shape(obj.re)


def size(obj):
    return np.size(obj.re)


def is_symmetry(obj):
    obj = to_dual(obj)
    return np.array_equal(obj.re, obj.re.T)


def is_symmetry_imag(obj):
    obj = to_dual(obj)
    return np.array_equal(obj.im, obj.im.T)


def is_symmetry_d(obj):
    return is_symmetry(obj) and is_symmetry_imag(obj)


def is_skew_symmetry(obj):
    obj = to_dual(obj)
    return np.array_equal(obj.re, - obj.re.T)


def is_skey_symmetry_imag(obj):
    obj = to_dual(obj)
    return np.array_equal(obj.im, - obj.im.T)


def is_skey_symmetry_d(obj):
    return is_skew_symmetry(obj) and is_skey_symmetry_imag(obj)


##########
# 配列生成関数 array-generate function
#######
def array(obj1, obj2=None, dtype=None, copy=True, order="K", subok=False, ndmin=0):
    if obj2 is None:
        return Dual(np.array(obj1, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    np.array(obj1, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    dtype=dtype)
    else:
        return Dual(np.array(obj1, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    np.array(obj2, copy=copy, order=order, subok=subok, ndmin=ndmin), \
                    dtype=dtype)


def arange(stop, start=None, step=None, stop_im=None, start_im=None, step_im=None,  dtype=None):
    if stop_im is None:
        stop_im = stop
    if start_im is None:
        start_im = start
    if step_im is None:
        step_im = step
    return Dual(np.arange(stop, start=start, step=step), \
                np.arange(stop_im, start=start_im, step=step_im), \
                dtype=dtype)


def linspace(start, stop, num=50, start_im=None, stop_im=None, num_im=None, \
             endpoint=True, retstep=False, dtype=None, axis=0):
    if start_im is None:
        startim = start
    if stop_im is None:
        stop_im = stop
    if num_im is None:
        num_im = num
    return Dual(np.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, \
                            axis=axis), \
                np.linspace(start_im, stop_im, num=num_im, endpoint=endpoint, retstep=retstep, \
                             axis=axis), \
                dtype=dtype)


def logspace(start, stop, num=50, start_im=None, stop_im=None, num_im=None, \
             endpoint=True, base=10.0, dtype=None, axis=0):
    if start_im is None:
        start_im = start
    if stop_im is None:
        stop_im = stop
    if num_im is None:
        num_im = num
    return Dual(np.logspace(start, stop, num=num, \
                            endpoint=endpoint, base=base, axis=axis), \
                np.logspace(start_im, stop_im, num=num_im, \
                            endpoint=endpoint, base=base, axis=axis), \
                dtype=dtype)


def mgrid(*xi, imag=1, **kwargs):
    x = np.meshgrid(*xi, **kwargs)
    return [Dual(xi, imag) for xi in x]


def ogrid(*xi, imag=1, **kwargs):
    x = np.ogrid(*xi, **kwargs)
    return [Dual(xi, imag) for xi in x]


def zeros(shape, dtype=np.float, order="C"):
    return Dual(np.zeros(shape, order=order), np.zeros(shape_im, order=order), dtype=dtype)


def zeros_like(a, dtype=None, order="K", subok=True):
    if is_dual(a):
        return Dual(np.zeros_like(a.re, order=order, subok=subok), \
                    np.zeros_like(a.re, order=order, subok=subok), dtype=dtype)
    else:
        return Dual(np.zeros_like(a, order=order, subok=subok), \
                    np.zeros_like(a, order=order, subok=subok), dtype=dtype)


def ones(shape, dtype=np.float, order="C"):
    return Dual(np.ones(shape, order=order), np.ones(shape, order=order), dtype=dtype)


def ones_like(a, dtype=None, order="K", subok=True):
    if is_dual(a):
        return Dual(np.ones_like(a.re, order=order, subok=subok), \
                    np.ones_like(a.re, order=order, subok=subok), dtype=dtype)
    else:
        return Dual(np.ones_like(a, order=order, subok=subok), \
                    np.ones_like(a, order=order, subok=subok), dtype=dtype)


def empty(shape, dtype=np.float, order="C"):
    return Dual(np.empty(shape, order=order), np.empty(shape, order=order), dtype=dtype)


def empty_like(prototype, dtype=None, order="K", subok=True, shape=None):
    if is_dual(prototype):
        return Dual(np.empty_like(prototype.re, order=order, subok=subok, shape=shape), \
                    np.empty_like(prototype.re, order=order, subok=subok, shape=shape), \
                    dtype=dtype)
    else:
        return Dual(np.empty_like(prototype, order=order, subok=subok, shape=shape), \
                    np.empty_like(prototype, order=order, subok=subok, shape=shape), \
                    dtype=dtype)


def full(shape, fill_value, fill_value_im=None, dtype=None, order="C"):
    if fill_value_im is None:
        fill_value_im = fill_value
    return Dual(np.full(shape, fill_value, order=order), \
                np.full(shape, fill_value_im, order=order), \
                dtype=dtype)


def full_like(a, fill_value, fill_value_im=None, dtype=None, order="K", subok=True, shape=None):
    if fill_value_im is None:
        fill_value_im = fill_value
    if is_dual(a):
        return Dual(np.full_like(a.re, fill_value, order=order, subok=subok, shape=shape), \
                    np.full_like(a.re, fill_value_im, order=order, subok=subok, shape=shape), \
                    dtype=dtype)
    else:
        return Dual(np.full_like(a, fill_value, order=order, subok=subok, shape=shape), \
                    np.full_like(a, fill_value_im, order=order, subok=subok, shape=shape), \
                    dtype=dtype)


def fill_diagonal(a, val, val_im=None, wrap=False):
    if val_im is None:
        val_im = val
    if is_dual(a):
        return Dual(np.fill_diagonal(a.re, val, wrap=wrap), \
                    np.fill_diagonal(a.im, val_im, wrap=wrap), \
                    dtype=dtype)
    else:
        return Dual(np.fill_diagonal(a, val, wrap=wrap), np.fill_diagonal(a, val_im, wrap=wrap), \
                    dtype=dtype)


def eye(N, M=None, k=0, dtype=np.float, order="C"):
    return Dual(np.eye(N, M=M, k=k, order=order), np.eye(N, M=M, k=k, order=order), \
                dtype=dtype)


def identity(n, dtype=None):
    return Dual(np.identity(n), np.identity(n), dtype=dtyoe)


def tile(A, reps, A_im=None):
    if is_dual(A):
        A_im = A.im
        A = A.re
    elif A_im is None:
        A_im = A

    return Dual(np.tile(A, reps), np.tile(A_im, reps))


def diag(v, v_im=None, k=0):
    if v_im is None:
        v_im = v
    return Dual(np.diag(v, k=k), np.diag(v_im, k=k))


def tri(N, M=None, k=0, dtype=float):
    return Dual(np.tri(N, M=M, k=k), np.tri(N, M=M, k=k), dtype=dtype)


def symmetry(obj, tri="l", k=0):
    if tri == "l":
        buf = tril(obj, k=k)
    elif tri == "u":
        buf = triu(obj, k=k)
    else:
        raise ValueError("Undefined argument value tri: {}".format(tri))
    return buf + buf.T - diag(buf.diagonal())


def skew_symmetry(obj, tri="l", k=0):
    if tri == "l":
        obj = tril(obj, k=k)
    elif tri == "u":
        obj = triu(obj, k=k)
    else:
        raise ValueError("Undefined argument value tri: {}".format(tri))
    return obj - obj.T


def copy(a, a_im=None, order="K"):
    if a_im is None:
        a_im = a
    return Dual(np.copy(a, order=order), np.copy(a_im, order=order))


def repeat(a, repeats, a_im=None, repeats_im=None, axis=None):
    if a_im is None:
        a_im = a
    if repeats_im is None:
        repeats_im = repeats
    return Dual(np.repeat(a, repeats, axis=axis), np.repeat(a_im, repeats_im, axis=axis))


def dcopy(obj, order="K"):
    obj = to_dual(obj)
    return Dual(np.copy(obj.re, order=order), np.copy(obj.im, order=order))


def where(condition, condition_im=None,  x=None, y=None):
    if condition_im is None:
        condition_im = condition
    if x is None:
        return (np.where(condition), np.where(condition_im))
    if y is None:
        return (np.where(condition), np.where(condition_im))
    return Dual(np.where(condition, x, y), np.where(condition_im, x, y))


##########
# 配列操作関数 operate-array functions
#######
def broadcast_to(obj, shape):
    obj = to_dual(obj)
    return Dual(np.broadcast_to(obj.re, shape), np.broadcast_to(obj.im, shape))


def _get_arrays(obj_list):
    re = []
    im = []
    for obj in obj_list:
        obj = to_dual(obj)
        re.append(obj.re)
        im.append(obj.im)
    return re, im


def _unite(obj_list, func, axis=None, out=None):
    re, im = _get_arrays(obj_list)
    if axis is None:
        if out is None:
            return Dual(func(re), func(im))
        else:
            return Dual(func(re, out=out), func(im, out=out))
    else:
        if out is None:
            return Dual(func(re, axis=axis), func(im, axis=axis))
        else:
            return Dual(func(re, axis=axis, out=out), func(im, axis=axis, out=out))


def concatenate(obj_list, axis=0, out=None):
    return _unite(obj_list, np.concatenate, axis=axis, out=out)


def stack(obj_list, axis=0, out=None):
    return _unite(obj_list, np.stack, axis=axis, out=out)


def vstack(obj_list):
    return _unite(obj_list, np.vstack)


def hstack(obj_list):
    return _unite(obj_list, np.hstack)


def dstack(obj_list):
    return _unite(obj_list, np.dstack)


def block(obj_list):
    def separate_into(obj_list):
        result_re = []
        result_im = []
        if isinstance(obj_list, list):
            for obj in obj_list:
                re, im = separate_into(obj)
                result_re.append(re)
                result_im.append(im)
            return result_re, result_im
        else:
            obj_list = to_dual(obj_list)
            return obj_list.re, obj_list.im
    re, im = separate_into(obj_list)
    return Dual(np.block(re), np.block(im))


def reshape(obj, newshape, order="C"):
    return to_dual(obj).reshape(newshape, order=order)


def squeeze(obj, axis=None):
    obj = to_dual(obj)
    return Dual(np.squeeze(obj.re, axis=axis), np.squeeze(obj.im, axis=axis))


def expand_dims(obj, axis):
    obj = to_dual(obj)
    return Dual(np.expand_dims(obj.re, axis), np.expand_dims(obj.im, axis))


def ravel(obj, order="C"):
    obj = to_dual(obj)
    return Dual(np.ravel(obj.re, order=order), np.ravel(obj.im, order=order))


def transpose(obj, axes=None):
    obj = to_dual(obj)
    return Dual(np.transpose(obj.re, axes=axes), np.transpose(obj.im, axes=axes))


def delete(obj, indices, axis=None):
    obj = to_dual(obj)
    return Dual(np.delete(obj.re, indices, axis=axis), np.delete(obj.im, indices, axis=axis))


def tril(obj, k=0):
    obj = to_dual(obj)
    return Dual(np.tril(obj.re, k=k), np.tril(obj.im, k=k))


def triu(obj, k=0):
    obj = to_dual(obj)
    return Dual(np.triu(obj.re, k=k), np.triu(obj.im, k=k))


def diag(obj, k=0):
    obj = to_dual(obj)
    return Dual(np.diag(obj.re, k=k), np.diag(obj.im, k=k))


def roll(obj, shift, axis=None):
    obj = to_dual(obj)
    return Dual(np.roll(obj.re, shift, axis=axis), np.roll(obj.im, shift, axis=axis))


def rollaxis(obj, axis, start=0):
    obj = to_dual(obj)
    return Dual(np.rollaxis(obj.re, axis, start=start), np.rollaxis(obj.im, axis, start=start)


##########
# ファイル入出力 input/output file
#######
def save(file, obj, allow_pickle=True, fix_imports=True):
    obj = to_dual(obj)
    np.save(file.replace(".npy", "_re.npy"), obj.re, \
            allow_pickle=allow_pickle, fix_imports=fix_imports)
    np.save(file.replace(".npy", "_im.npy"), obj.im, \
            allow_pickle=allow_pickle, fix_imports=fix_imports)


def savez(file, *args, **kwds):
    _kwds = []
    if len(args):
        i = 0
        for obj in args:
            obj = to_dual(obj)
            _kwds.append("arr_{}_re".format(i))
            _kwds.append(obj.re)
            _kwds.append("arr_{}_im".format(i))
            _kwds.append(obj.im)
            i += 1
    if len(kwds):
        for key in kwds:
            obj = to_dual(kwds[key])
            _kwds.append(str(key) + "_re")
            _kwds.append(obj.re)
            _kwds.append(str(key) + "_im")
            _kwds.append(obj.im)

    np.savez(file, **dict(zip(_kwrds[0::2], _kwds[1::2])))


def savetxt(fname, obj, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', \
            comments='# ', encoding=None):
    obj = to_dual(obj)
    fname, ext = os.path.splitext(fname)
    np.savetxt(fname + "_re" + ext, obj.re, fmt=fmt, delimiter=delimiter, newline=newline, \
               header=header, footer=footer, comments=comments, encoding=encoding)
    np.savetxt(fname + "_im" + ext, obj.im, fmt=fmt, delimiter=delimiter, newline=newline, \
               header=header, footer=footer, comments=comments, encoding=encoding)


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    if ".npy" in file:
        return  Dual(np.load(file.replace(".npy", "_re.npy"), mmap_mode=mmap_mode, \
                             allow_pickle=allow_pickle, fix_imports=fix_imports, \
                             encoding=encoding), \
                     np.load(file.replace(".npy", "_im.npy"), mmap_mode=mmap_mode, \
                             allow_pickle=allow_pickle, fix_imports=fix_imports, \
                             encoding=encoding))
    elif ".npz" in file:
        buf = np.load(file, mmap_mode=mmap_mode, \
                      allow_pickle=allow_pickle, fix_imports=fix_imports, \
                      encoding=encoding)
        _args = []
        _kwds = []
        for key in buf:
            if "arr_" in key:
                _args.append(buf[key])
            else:
                _kwds.append(key)
                _kwds.append(buf[key])

        args = []
        kwds = []
        for i in range(0, len(_args), 2):
            args.append(Dual(_args[i], _args[i + 1]))

        for i in range(0, len(_kwds), 4):
            kwds.append(_kwds[i].replace("_re", ""))
            kwds.append(Dual(_kwds[i + 1], _kwds[i + 3]))

        return tuple(args), dict(zip(kwds[0::2], kwds[1::2]))


def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, \
            usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None):
    fname, ext = o?!?jedi=0, s.path.splitext(fname)?!? (fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None) ?!?jedi?!?
    re = np.loadtxt(fname + "_re" + ext, dtype=dtype, comments=comments, delimiter=delimiter, \
                    converters=converters, skiprows=skiprows, usecols=usecols, unpack=unpack, \
                    ndim=ndim, encoding=encoding, max_rows=max_rows)
    im = np.loadtxt(fname + "_im" + ext, dtype=dtype, comments=comments, delimiter=delimiter, \
                    converters=converters, skiprows=skiprows, usecols=usecols, unpack=unpack, \
                    ndim=ndim, encoding=encoding, max_rows=max_rows)
    return Dual(re, im)

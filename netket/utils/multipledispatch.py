from plum import dispatch
from numbers import Number

import numpy as np


@dispatch
def add(A, B):
    return NotImplemented


@dispatch
def iadd(A, B):
    return NotImplemented


@dispatch
def sub(A, B):
    return add(A, -B)


@dispatch
def isub(A, B):
    return iadd(A, -B)


@dispatch
def mul(A, B):
    return NotImplemented


@dispatch
def imul(A, B):
    return NotImplemented


@dispatch
def imatmul(A, B):
    return NotImplemented


@dispatch
def dtype(x: Number):
    return type(x)


@dispatch
def dtype(x: np.ndarray):
    return x.dtype

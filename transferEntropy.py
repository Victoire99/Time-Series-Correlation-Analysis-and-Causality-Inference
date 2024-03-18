'''
Author: KEWEI ZHANG
Date: 2024-02-01 13:12:18
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-20 16:12:52
FilePath: \WorkNote\Term Conclusion\transferEntropy.py
Description: transfer entropy functions

'''

# baqsic packages
import pandas as pd
import numpy as np
import os
import operator
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import IPython
import IPython.display
from IPython.display import display
from collections import OrderedDict

# pyinform packages
from ctypes import byref, c_int, c_ulong, c_double, POINTER
import pyinform
from pyinform import _inform
from pyinform.error import ErrorCode, error_guard




#! Transfer Entropy Functon
def transfer_entropy(source, target, k, condition=None, local=False):
    """
    Compute the local or average transfer entropy from one time series to
    another with target history length *k*. Optionally, time series can be
    provided against which to *condition*.
    :param source: the source time series
    :type source: sequence or ``numpy.ndarray``
    :param target: the target time series
    :type target: sequence or ``numpy.ndarray``
    :param int k: the history length
    :param condition: time series of any conditions
    :type condition: sequence or ``numpy.ndarray``
    :param bool local: compute the local transfer entropy
    :returns: the average or local transfer entropy
    :rtype: float or ``numpy.ndarray``
    :raises ValueError: if the time series have different shapes
    :raises ValueError: if either time series has no initial conditions
    :raises ValueError: if either time series is greater than 2-D
    :raises InformError: if an error occurs within the ``inform`` C call
    """
    ys = np.ascontiguousarray(source, np.int32)
    xs = np.ascontiguousarray(target, np.int32)
    cs = np.ascontiguousarray(condition, np.int32) if condition is not None else None
 
    if xs.shape != ys.shape:
        raise ValueError("source and target timeseries are different shapes")
    elif xs.ndim > 2:
        raise ValueError("source and target have too great a dimension; must be 2 or less")
 
    if cs is None:
        pass
    elif cs.ndim == 1 and cs.shape != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim == 2 and xs.ndim == 1 and cs.shape[1:] != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim == 2 and xs.ndim == 2 and cs.shape != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim == 3 and cs.shape[1:] != xs.shape:
        raise ValueError("condition has a shape that's inconsistent with the source and target")
    elif cs.ndim > 3:
        raise ValueError("condition has too great a dimension; must be 3 or less")
 
    ydata = ys.ctypes.data_as(POINTER(c_int))
    xdata = xs.ctypes.data_as(POINTER(c_int))
    cdata = cs.ctypes.data_as(POINTER(c_int)) if cs is not None else None
 
    if cs is None:
        b = max(2, max(np.amax(xs), np.amax(ys)) + 1)
    else:
        b = max(2, max(np.amax(xs), np.amax(ys), np.amax(cs)) + 1)
 
    if cs is None:
        z = 0
    elif cs.ndim == 1 or (cs.ndim == 2 and xs.ndim == 2):
        z = 1
    elif cs.ndim == 3 or (cs.ndim == 2 and xs.ndim == 1):
        z = cs.shape[0]
    else:
        raise RuntimeError("unexpected state: condition and source are inconsistent shapes")
 
    if xs.ndim == 1:
        n, m = 1, xs.shape[0]
    else:
        n, m = xs.shape
 
    e = ErrorCode(0)
 
    if local is True:
        q = max(0, m - k)
        te = np.empty((n, q), dtype=np.float64)
        out = te.ctypes.data_as(POINTER(c_double))
        _local_transfer_entropy(ydata, xdata, cdata, c_ulong(z), c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), out, byref(e))
    else:
        te = _transfer_entropy(ydata, xdata, cdata, c_ulong(z), c_ulong(n), c_ulong(m), c_int(b), c_ulong(k), byref(e))
 
    error_guard(e)
 
    return te


 
_transfer_entropy = _inform.inform_transfer_entropy
_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_ulong, c_int, c_ulong, POINTER(c_int)]
_transfer_entropy.restype = c_double
 
_local_transfer_entropy = _inform.inform_local_transfer_entropy
_local_transfer_entropy.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), c_ulong, c_ulong, c_ulong, c_int, c_ulong, POINTER(c_double), POINTER(c_int)]
_local_transfer_entropy.restype = POINTER(c_double)


def trans_en_x(df):
#! the distribution of the state is more important than the actual value in information theory
#* 选中列为因，其他为果 
    t_dic = {}
    x = df.iloc[:,0]
    # transfer negative value to positive
    coal_x, a = pyinform.utils.coalesce_series(x)
    for i in range(1, len(df.columns)):
        y = df.iloc[:,i]
        coal_y, b = pyinform.utils.coalesce_series(y)
        a = transfer_entropy(coal_x, coal_y, k=2)
        t_dic[df.columns[i]] = a
    return t_dic

def trans_en_y(df):
#! the distribution of the state is more important than the actual value in information theory
#* 选中列为果，其他为因
    t_dic = {}
    x = df.iloc[:,0]
    # transfer negative value to positive
    coal_x, a = pyinform.utils.coalesce_series(x)
    for i in range(1, len(df.columns)):
        y = df.iloc[:,i]
        coal_y, b = pyinform.utils.coalesce_series(y)
        a = transfer_entropy(coal_y, coal_x, k=2)
        t_dic[df.columns[i]] = a
    return t_dic
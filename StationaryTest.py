'''
Author: KEWEI ZHANG
Date: 2023-12-01 12:19:42
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-12 10:25:34
FilePath: \WorkNote\Term Conclusion\StationaryTest.py
Description: 

'''

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller,kpss,coint

from datetime import datetime

from functions import *


# stationary test
# adf test
def adf_test(df):
    result = adfuller(df.values)
    if result[1] <= 0.05:
        print(f'{df.name} is stationary')
    else:
        print(f'{df.name} is non-stationary')

#kspp test
def kpss_test(df):    
    statistic, p_value, n_lags, critical_values = kpss(df.values)
    if p_value <= 0.05:
        print(f'{df.name} is non-stationary')
    else:
        print(f'{df.name} is stationary')


def station_test(variables,df):
    for item in variables:
        print('ADF Test of: ' + item)
        adf_test(df[item])
        print('KPSS Test of: ' + item)
        kpss_test(df[item])


# Engel Granger test

def en_gr_test(df):
    # list stored the coningerated variables 
    coin_liz = []
    # dict stored all the coningerated coefficients 
    coin_dct = {}
    for i in range(1, len(df.columns)):
        p1 = df.iloc[:,0]
        p2 = df.iloc[:,i]
        x = np.log(p2)
        y = np.log(p1)
        coin = coint(x,y)
        # save the non-cointegrated variable
        if coin[0] < -100:
            coin_liz.append(i)
        # print results
        now = datetime.now()
        coin_dct[df.columns[i]] = coin
        with open('output_cointegration.txt', 'a') as f:
            print("Running Time: ", now,file = f)
            print("The main variable is: ", df.columns[0], file=f)
            print("The second variable is: ", df.columns[i], file=f)
            print("The p-value of the cointegration test is: ", coin[0], file=f)
            print("The threshold array is: ", coin[2], file=f)
    print("The coingeration variables are: ",df.columns[coin_liz])
    return coin_liz, coin_dct

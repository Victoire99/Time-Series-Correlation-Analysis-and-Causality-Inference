'''
Author: KEWEI ZHANG
Date: 2024-02-06 14:37:05
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-12 15:42:30
FilePath: \WorkNote\Term Conclusion\CausalityInference.py
Description: 

'''



import numpy as np
import pandas as pd
import math
from datetime import datetime
import matplotlib.pyplot as plt


from statsmodels.tsa.stattools import grangercausalitytests
from numpy.linalg import LinAlgError
from transferEntropy import *

from math_functions import *

import tensorflow as tf
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler


# granger causality test
def granger_causal_test(df,test='ssr_chi2test',maxlag=15, verbose=False):
    x = df.iloc[:,0]
    gran_liz= []
    gran_dct = {}
    for i in range(1,len(df.columns)):
        try: 
            y = df.iloc[:,i]
            data = np.column_stack([y,x])
            gc_res = grangercausalitytests(data,  maxlag=maxlag, verbose=False)
            p_values = [round(gc_res[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {i}, X = {0}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            if min_p_value < 0.05:
                gran_liz.append(i)
        except LinAlgError : 
            pass
        # print results
        gran_dct[df.columns[i]] = min_p_value
        
        with open('output_granger.txt', 'a') as f:
            print("The x variable is: ", df.columns[0], file=f)
            print("The y variable is: ", df.columns[i], file=f)
            print("The p-value of the granger causality test is: ", min_p_value, file=f)
    print(f"The granger causing variables to {df.columns[0]} are: ",df.columns[gran_liz])

    return gran_liz, gran_dct



# transfer entropy
def trans_en_test(df):
    columns = df.columns
    scaler = RobustScaler()
    df_t = scaler.fit_transform(df)
    df_t = pd.DataFrame(df_t, columns=columns)

    x_dic = trans_en_x(df_t)
    y_dic = trans_en_y(df_t)

    # x->y
    te_x_dic = {}
    text_x = df_t.columns[0]
    for i in range(len(x_dic.keys())):
        text_y = list(x_dic.keys())[i]   
        x = x_dic[text_y]
        y = y_dic[text_y]
        with open('output_transferEntropy.txt', 'a') as f:
            if x > y:
                te_x_dic[text_y] = -x
                print(f"The transfer entropy from {text_x} to {text_y} is: ", -x, file=f)
            else:
                te_x_dic[text_y] = y
                print(f"The transfer entropy from {text_x} to {text_y}  is: ", y, file=f)
    print("hint: the symbol means the direction of the transfer entropy, the negative means x->y, the positive means y->x")
    sorted_dct = OrderedDict(sorted(te_x_dic.items(), key=lambda item: item[1],reverse=True))
    y_dct = {k:v for k,v in sorted_dct.items()}  # WORKAROUND
   
    feature_plot(y_dct,text_x,title = 'Transfer Entropy')

    return y_dct
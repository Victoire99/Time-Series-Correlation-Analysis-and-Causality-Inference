'''
Author: KEWEI ZHANG
Date: 2023-12-11 15:40:19
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-06 12:40:15
FilePath: \WorkNote\Term Conclusion\grangers.py
Description: graner causality test

'''
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose

from functions import *
from math_functions import *
from method import *

from numpy.linalg import LinAlgError

def grangers_causation_matrix(data, variables, maxlag=15, test='ssr_chi2test', verbose=False): 
    '''Check Granger Causality of all possible combinations of the Time series.'''
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            try: 
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
            except LinAlgError : 
                pass
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    df_dict = df.to_dict()
    return df_dict


def mergeList(df):
    '''merge the list of granger causality test'''
    temp = []
    for i in range(len(df.columns)):
        for j in range(len(df.index)):
            if 0 < df.iloc[i,j] < 0.05:
                name = df.columns[i] + '->' + df.index[j]
                temp.append(name)
    return temp

def mergeDf(dic):
    '''merge the dataframe of granger causality test'''
    temp = []
    for key in dic.keys():
        df = pd.DataFrame(dic[key])
        temp.append(mergeList(df))
    return temp

def station_process(stat_dic, variables):
    '''stationary process'''
    results_dic = {}
    

    for variable  in variables:
        variable_dic = {}

        difference(stat_dic,variable,variable_dic)

        moving_average(stat_dic,variable,variable_dic,1,'ma_1')
        moving_average(stat_dic,variable,variable_dic,2,'ma_2')
        moving_average(stat_dic,variable,variable_dic,3,'ma_3')
        moving_average(stat_dic,variable,variable_dic,4,'ma_4')

        moving_median(stat_dic,variable,variable_dic,1,'mm_1')
        moving_median(stat_dic,variable,variable_dic,2,'mm_2')
        moving_median(stat_dic,variable,variable_dic,3,'mm_3')
        moving_median(stat_dic,variable,variable_dic,4,'mm_4')

        results_dic[variable] = variable_dic

    return results_dic

def granger_Cal(available_list,variables,results_dic):
    '''granger causality test process '''
    re_dict = {}
    for i in range(len(available_list)):
        diff_dic = {}
        key = available_list[i]
        for item in variables:
            diff_dic[item] = results_dic[item][key]
        df_transformed = pd.DataFrame(diff_dic)

        df_transformed = df_transformed.loc[:, (df_transformed != df_transformed.iloc[0]).any()]
        df_transformed = df_transformed.dropna(how='any')

        dic = grangers_causation_matrix(df_transformed, variables = df_transformed.columns)
        re_dict[key] = dic
        
    return re_dict

def dropDup(merge_list):
    '''drop the duplicate item in the list'''
    merged_items = []
    for sublist in merge_list:
        merged_items.extend(sublist)
    new_list = list(set(merged_items))
    new_list.sort()

    return new_list
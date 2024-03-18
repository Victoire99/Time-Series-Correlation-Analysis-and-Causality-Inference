'''
Author: KEWEI ZHANG
Date: 2024-02-06 14:36:18
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-12 15:21:21
FilePath: \WorkNote\Term Conclusion\CorrelationAnalysis.py
Description: Correlation Analysis Realted Functions

'''



import numpy as np
import pandas as pd
import math
from datetime import datetime

from scipy import stats
from statsmodels.tsa.stattools import coint

from math_functions import *



# Pearson linear correlation
def pearson_corr_test(df):
    x = df.iloc[:,0]
    pr_coe_dct = {}
    for i in range(1,len(df.columns)):
        y = df.iloc[:,i]
        corr_matrix = np.corrcoef(x,y)
        corr_coe = corr_matrix[0, 1]
        pr_coe_dct[df.columns[i]] = corr_coe
        with open('output_pearson.txt', 'a') as f:
                print("The x variable is: ", df.columns[0], file=f)
                print("The y variable is: ", df.columns[i], file=f)
                print("The Pearson linear correlation is: ", corr_coe, file=f)
    # plot
    feature_plot(pr_coe_dct,'% Processor Time_RapidResponse','Pearson Linear Correlation Coefficient') 

    return pr_coe_dct
    

# spearmanr level correlation
def spearman_corr_test(df):
    x = df.iloc[:,0]
    spear_coe_dct = {}
    for i in range(1,len(df.columns)):
        y = df.iloc[:,i]
        corr, p = stats.spearmanr(x, y)
        spear_coe_dct[df.columns[i]] = {"correlation":corr, "p-value":p}
        with open('output_spearman.txt', 'a') as f:
                print("The x variable is: ", df.columns[0], file=f)
                print("The y variable is: ", df.columns[i], file=f)
                print("The spearman correlation is: ", corr, file=f)
                print("The p-value is: ", p, file=f)
    
    # plot
    spe_cols = list(spear_coe_dct.keys())
    spe_coe_dct = {}
    spe_p_dct = {}
    for i in range(len(spe_cols)):
        spe_coe_dct[spe_cols[i]] = spear_coe_dct[spe_cols[i]]['correlation']
        spe_p_dct[spe_cols[i]] = spear_coe_dct[spe_cols[i]]['p-value']
    
    feature_plot(spe_coe_dct,'% Processor Time_RapidResponse','Spearman Level Coefficient') 
    feature_plot(spe_p_dct,'% Processor Time_RapidResponse','Spearman Level Coefficient p-value') 

    return spear_coe_dct
    

# kendall level correlation
def kendall_corr_test(df):
    x = df.iloc[:,0]
    kendall_coe_dct = {}
    for i in range(1,len(df.columns)):
        y = df.iloc[:,i]
        corr, p = stats.kendalltau(x, y)
        kendall_coe_dct[df.columns[i]] = {"correlation":corr, "p-value":p}
        with open('output_kendall.txt', 'a') as f:
                print("The x variable is: ", df.columns[0], file=f)
                print("The y variable is: ", df.columns[i], file=f)
                print("The kendall correlation is: ", corr, file=f)
                print("The p-value is: ", p, file=f)

    # plot
    ken_cols = list(kendall_coe_dct.keys())
    ken_coe_dct = {}
    ken_p_dct = {}
    for i in range(len(ken_cols)):
        ken_coe_dct[ken_cols[i]] = kendall_coe_dct[ken_cols[i]]['correlation']
        ken_p_dct[ken_cols[i]] = kendall_coe_dct[ken_cols[i]]['p-value']
    
    feature_plot(ken_coe_dct,'% Processor Time_RapidResponse','Kendall Level Coefficient') 
    feature_plot(ken_p_dct,'% Processor Time_RapidResponse','Kendall Level Coefficient p-value') 

    return ken_coe_dct
    
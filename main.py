'''
Author: KEWEI ZHANG
Date: 2024-02-12 09:47:07
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-20 13:40:44
FilePath: \WorkNote\Term Conclusion\main.py
Description: main running function

'''

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


sys.path.append(r"C:\WorkNote\datadogCorr")
sys.path.append(r"C:\WorkNote\DL")

from PerformanceCounter import *
from functions import *
from StationaryTest import *
from math_functions import *
from transferEntropy import *
from CorrelationAnalysis import *
from CausalityInference import *
from neuralNetwork import *

import warnings       
warnings.filterwarnings ("ignore")

if __name__ == '__main__':
    print('This is the main function')

    #! selected variables: processor_time_RapidResponse

    sp_list = ['Users logged in','Total query execution time','Total data query executions',
    'Queries being executed','Queries open','Lock wait time (ms/sec)','DataChange submit rate','% Processor Time']
    pc_list_num = 0
    proce_time = PC(pc_list_num,sp_list)

    dct = proce_time.sp_dct
    mc_list = proce_time.machine_list
    sp_var = proce_time.sp_variables
    sp = sp_var[0]

    # already select the machine num here 
    # the first machine does not have the processor time metrics, all 0
    machine_list_num = 1
    scaler = RobustScaler()
    df,df_dct = display_df_pc(dct, mc_list, machine_list_num ,sp_var)

    #! cointegration test
    # remove all 0 col
    # engle-granger test
    df = df.loc[:, (df != 0).any(axis=0)]
    coin_liz, coin_dct = en_gr_test(df)

    #! data visualization
    # view all the data
    fig = plt.figure( )
    df.plot(subplots=True,figsize=(30,25))
    plt.show()
    # visualize the unstationary data
    uncoin_df = df.drop(columns = df.columns[coin_liz])
    uncoin_df = uncoin_df.drop(columns = ['% Processor Time_RapidResponse'])
    fig = plt.figure( )
    uncoin_df.plot(subplots=True,figsize=(20,15))
    plt.legend(loc='upper right')
    plt.show()

    #! stationary transformation
    # diff period = 4 (per hour)
    diff = uncoin_df.diff(4).dropna()
    #df[df.columns[coin_liz]]
    mod_df = pd.concat([df['% Processor Time_RapidResponse'],df[df.columns[coin_liz]]],axis = 1)
    mod_df = pd.concat([mod_df,diff],axis = 1)
    mod_df.fillna(0, inplace=True)

    #! correlation test
    pearson_coe_dct = pearson_corr_test(mod_df)

    spear_coe_dct = spearman_corr_test(mod_df)

    ken_coe_dct = kendall_corr_test(mod_df)

    #! causality test
    gran_liz,gran_dct = granger_causal_test(mod_df)

    #transfer entropy
    trans_en_dct = trans_en_test(mod_df)


    #! LSTM                                                                                                                                                                                                                                                     LSTM
    drop_col_list = []
    train_df, val_df, test_df = train_test_val_split(mod_df)
    train_df, val_df, test_df, mod_df ,variables = preprocess_df(train_df, val_df, test_df, mod_df, drop_col_list)

    output_path  = 'C:/WorkNote/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path )

    nn_model_vis(variables,train_df, val_df, test_df,output_path)
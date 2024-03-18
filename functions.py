'''
Author: KEWEI ZHANG
Date: 2024-01-16 12:42:53
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-20 15:57:27
FilePath: \WorkNote\Term Conclusion\functions.py
Description: all functions for LSTM 

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import layers
from tqdm import tqdm
import IPython
import IPython.display
from Window import WindowGenerator

MAX_EPOCHS = 50

def read_data_pc(path, pc_list,num):
    '''
    description: read the data from performance counter
    param path {*}: file path
    param pc_list {list}: list stored all excel files name
    param num {int}: # of file
    return 
    df_dic {dict}: dict stored all data
    machine_list {list}: list stored all machine names
    variables {list}: list stored all variables
    time_list {list}: time list
    '''

    df = pd.read_excel(path + '/' + pc_list[num])

    # let the df ordered by the column 'Performance Counter name'   
    df = df.sort_values(by=['Performance Counter Name','Start Time','Performance Counter Category Name','Machine'])
    # ! 这里取数值用的是Max Value，如果要改成其他的可以在这里改, 但是要注意后面的代码也要改
    # ! ['Avg Value','Max Value','Min Value'] 三选一
    df.drop(['Log Id','End Time','Count','Avg Value','Min Value'], axis=1, inplace=True)

    # fill the nan with ' '
    df = df.fillna(" ")

    # two servers for one customer
    machine_list = list(df['Machine'].drop_duplicates())

    # time list 
    # *只存时间序列一次，之前一直给每个键都存储了一次，浪费空间
    time_list = np.array(df['Start Time'].drop_duplicates())

    df_dic = {}

    # iteratively store the data into the dict
    for i in range(len(machine_list)):
        df_temp = df[df['Machine'] == machine_list[i]]
        grouped = df_temp.groupby(['Performance Counter Category Name', 'Performance Counter Name','Performance Counter Instance Name'])

        # 你如果要把dict当作内容存入另一个dict那你必须这样单独先生成然后再存入，直接存入会失败不知道为啥
        temp_dict = {}
        for key, group in grouped:
            #time_series = group['Start Time'].values
            # ! 这里取数值用的是Max Value，前面改了的话这里也要改
            value_series = group['Max Value'].values
            name = key[1] + '_' + key[2]
            temp_dict[name] = np.array(value_series)
            variables = list(temp_dict.keys())
        
        df_dic[machine_list[i]] = temp_dict
        
    #time_list = df_dic[machine_list[0]][variables[0]]['time']   

    return df_dic, machine_list, variables, time_list




def read_data_pc_rr(path, pc_list,num):
    '''
    description: read the data from performance counter of RapidResponse Server
    param path {*}: file path
    param pc_list {list}: list stored all excel files name
    param num {int}: # of file
    return 
    df_dic {dict}: dict stored all data
    machine_list {list}: list stored all machine names
    variables {list}: list stored all variables
    time_list {list}: time list
    '''    
    df = pd.read_excel(path + '/' + pc_list[num])

    # let the df ordered by the column 'Performance Counter name'  
    df = df[df['Performance Counter Category Name'].isin(['RapidResponse Server', 'Process'])]
    df = df.sort_values(by=['Performance Counter Name','Start Time','Performance Counter Category Name','Machine'])
    # ! 这里取数值用的是Max Value，如果要改成其他的可以在这里改, 但是要注意后面的代码也要改
    # ! ['Avg Value','Max Value','Min Value'] 三选一
    df.drop(['Log Id','End Time','Count','Avg Value','Min Value'], axis=1, inplace=True)

    # fill the nan with ' '
    df = df.fillna(" ")

    # two servers for one customer
    machine_list = list(df['Machine'].drop_duplicates())

    # time list 
    time_list = np.array(df['Start Time'].drop_duplicates())


    df_dic = {}
    
    for i in range(len(machine_list)):
        df_temp = df[df['Machine'] == machine_list[i]]
        grouped = df_temp.groupby(['Performance Counter Category Name', 'Performance Counter Name','Performance Counter Instance Name'])

        temp_dict = {}
        for key, group in grouped:
            #time_series = group['Start Time'].values
            # ! 这里取数值用的是Max Value，前面改了的话这里也要改
            value_series = group['Max Value'].values
            name = key[1] + '_' + key[2]
            temp_dict[name] = {'value': np.array(value_series)}
            variables = list(temp_dict.keys())
        
        df_dic[machine_list[i]] = temp_dict
        
    #time_list = df_dic[machine_list[0]][variables[0]]['time']   

    return df_dic, machine_list, variables, time_list

def read_data_hb(path, hb_list,num):
    '''
    description: read the data from heartbeat
    param path {*}: file path
    param hb_list {*}: list stored all excel files name
    param num {int}: # of file
    return 
    df_dic {dict}: dict stored all data
    variables {list}: list stored all variables
    time_list {list}: time list
    '''

    df = pd.read_excel(path + '/' + hb_list[num])
    df.columns = df.loc[1]
    df = df.drop([0,1])

    # ! 目前我们只选取了一部分的数据，如果需要其他的数据可以在这里改
    # * trash_list: useless columns, we don't need them
    # * transfer_list: transfer the columns type from string to float
    # * text_list: text columns, we don't need them
    trash_list = ['EarliestPinnedViewCreationTime','Server Log Id','Timestamp']
    transfer_list = ['ActiveQueriesPercent','Collision1','Collision2','Collision3','Collision4','Collision5','MxWt1 (s)','MxWt2 (s)','MxWt3 (s)','MxWt4 (s)','MxWt5 (s)']
    text_list = ['SlowLock1','SlowLock2','SlowLock3','SlowLock4','SlowLock5']

    str_list = ['Page1(GB)','Page2(GB)', 'Page3(GB)', 'Page4(GB)', 'Page5(GB)', 'Pool1(GB)','Pool2(GB)','Pool3(GB)','Pool4(GB)','Pool5(GB)' ]

    df.reset_index(drop=True, inplace=True)
    time_list = np.array(df['Timestamp'])

    # useless columns
    for i in range(len(trash_list)):
        df = df.drop(trash_list[i], axis=1)
    for i in range(len(text_list)):
        df = df.drop(text_list[i], axis=1)

    # transfer the columns type from string to float
    for i in range(len(transfer_list)):
        df = df.drop(transfer_list[i], axis=1)

    df = df.loc[:, (df != df.iloc[0]).any()]

    for i in range(len(str_list)):
        for j in range(len(df)):
            df[str_list[i]][j] =  df[str_list[i]][j].split(':')[-1]

    df = df.astype(float)

    variables = list(df.columns[1:])
    df_dic = {}
    for x in variables:
        df_dic[x] = {'time': time_list, 'value': np.array(df[x])}
    
    return df_dic, variables, time_list


def dis_df(df):
    '''
    description: plot the whole data visualization, using violin plot
    param df {dataframe}: dataframe
    return {*} violin plot
    '''
    # observe the distribution of the metrics
    #df_std = pd.DataFrame(scaler1.fit_transform(df))
    df_mean = df.mean()
    df_std = df.std()
    df_std = (df - df_mean) / df_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()

def normalize_df(df,scaler):
    '''
    description: normalize the dataframe 
    param df {*}: input dataframe
    param scaler {*}: scaler
    return {*} normalized dataframe
    '''
    #train_df = (train_df - train_mean) / train_std
    #val_df = (val_df - train_mean) / train_std
    #test_df = (test_df - train_mean) / train_std
    # robust scaler
    columns = df.columns
    df = scaler.fit_transform(df)

    df = pd.DataFrame(df, columns=columns)

    return df

def display_df_pc(df_dic, machine_list,machine_num ,variables):
    '''
    description: step1
    param df_dic {*}:
    param machine_list {*}:
    param machine_num {*}:
    param variables {*}:
    param scaler {*}:
    return {dataframe}: four dataframes
    '''

    val_dic = {}
    for j in range(len(variables)):
        val_dic[variables[j]] = df_dic[machine_list[machine_num]][variables[j]] 

    df = pd.DataFrame(val_dic)

    # remove the variables which are not very related
    df = df.drop(columns=variables[2:5])
    
    # observe the distribution of the metrics
    dis_df(df)

    return df,val_dic
    
def train_test_val_split(df):
    #var =df.columns
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    scaler = RobustScaler()
    train_df = normalize_df(train_df,scaler)
    val_df = normalize_df(val_df,scaler)
    test_df = normalize_df(test_df,scaler)

    train_df = pd.DataFrame(train_df, columns=df.columns)
    val_df = pd.DataFrame(val_df, columns=df.columns)
    test_df = pd.DataFrame(test_df, columns=df.columns)
    
    return train_df,val_df,test_df



def display_df_hb(df_dic,variables,scaler):
    '''
    description: step1
    param df_dic {*}:
    param machine_list {*}:
    param machine_num {*}:
    param variables {*}:
    param scaler {*}:
    return {dataframe}: four dataframes
    '''
    
    val_dic = {} 
    for j in range(len(variables)):
        val_dic[variables[j]] = df_dic[variables[j]]['value']   
    df = pd.DataFrame(val_dic)


    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    #num_features = df.shape[1]
    #column_indices = {name: i for i, name in enumerate(df.columns)}

    # normalize the data
    # here we can change the way to normalize the data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = normalize_df(train_df,scaler)
    val_df = normalize_df(val_df,scaler)
    test_df = normalize_df(test_df,scaler)

    train_df = pd.DataFrame(train_df, columns=df.columns)
    val_df = pd.DataFrame(val_df, columns=df.columns)
    test_df = pd.DataFrame(test_df, columns=df.columns)
    
    # observe the distribution of the metrics
    dis_df(df, train_mean,train_std)

    return train_df, val_df, test_df, df




def preprocess_df(train_df, val_df, test_df, df, drop_col_list):
    '''
    description: drop the blank cloumns by violin plot
    param train_df {*}:
    param val_df {*}:
    param test_df {*}:
    param df {*}:
    param drop_col_list {*}:
    return {*}
    '''
    
    train_df = train_df.drop(columns=drop_col_list)
    val_df = val_df.drop(columns=drop_col_list)
    test_df = test_df.drop(columns=drop_col_list)
    df = df.drop(columns=drop_col_list)
    variables = df.columns

    #timegap = (time_list[1] - time_list[0])/ np.timedelta64(1, 's')
    #timegap = timegap.astype(int)

    return train_df, val_df, test_df, df, variables

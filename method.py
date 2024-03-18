'''
Author: KEWEI ZHANG
Date: 2023-11-28 14:45:33
LastEditTime: 2024-02-06 12:39:01
FilePath: \WorkNote\Term Conclusion\method.py
Description: stationary transformation methods

'''

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

from functions import *
from math_functions import *

COL_NAME = "query"
COL_TIME_NAME = "time"
COL_VALUE_NAME = "value"
TRAIN_SIZE = 0.8




# 当前数据为前一天/周的数据
# 如果不存在前一天/周的日期（起始日、周），则保持不变
# 输入： dic: 大字典名称； var:大字典中的变量名名称 var_dict:该变量存储所有detector的字典，预先定义

# last day
def diff_fun_ld(dic,var):
 
    time_list, val_list,date_points_num,train_size,test,train = config(dic,var)
    
    date_points_num = int(date_points_num) * 1
    # current value is the last day value
    # the first day's value stay, do not change
    val_list_diff = np.concatenate((val_list[:date_points_num],val_list[:-date_points_num]),axis=0)
    d = {'diff_ld':val_list_diff}
    m, r = perfTest(val_list,val_list_diff)
    per = {'mape_ld':m, 'rmse_ld':r}
    return d, per

#last week
def diff_fun_lw(dic,var):

    time_list, val_list,date_points_num,train_size,test,train = config(dic,var)

    date_points_num = int(date_points_num) * 7
    # current value is the last day value
    # the first day's value stay, do not change
    val_list_diff = np.concatenate((val_list[:date_points_num],val_list[:-date_points_num]),axis=0)
    d = {'diff_lw':val_list_diff}
    m, r = perfTest(val_list,val_list_diff)
    per = {'mape_lw':m, 'rmse_lw':r}
    return d, per

# history average
def ha_fun(dic,var,window,key):
    
    time_list, val_list,date_points_num,train_size,train,test = config(dic,var)

    his_avg=pd.Series(val_list).rolling(window).mean()
    his_avg = np.asarray(his_avg)
    his_ma = {key:his_avg}
    return his_ma


# history median
def hm_fun(dic,var,window,key):
    
    time_list, val_list,date_points_num,train_size,train,test= config(dic,var)

    his_me=pd.Series(val_list).rolling(window).median()
    his_me = np.asarray(his_me)
    his_mm = {key:his_me}
    return his_mm

#holt-winters
def hw_fun(dic,var):
	
	time_list, val_list,date_points_num,train_size,train,test = config(dic,var)

	periods_list = [date_points_num, 7*date_points_num]
	char_list =  ['add', 'mul', None]
	para_list = [0.2,0.4,0.6,0.8]

	# 指定参数
	temp = []
	for seasonal_periods in periods_list:
		for smoothing_level in para_list:
			for smoothing_trend in para_list:
				for smoothing_seasonal in para_list:
					model = ExponentialSmoothing(np.asarray(train), seasonal_periods=seasonal_periods, trend='mul',seasonal='mul')
					fit1 = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, smoothing_seasonal=smoothing_seasonal)
					val_list_hw = fit1.forecast(len(val_list))
					temp.append(val_list_hw)
	#不指定参数，用optimized 
	for seasonal_periods in periods_list:
		model = ExponentialSmoothing(np.asarray(train), seasonal_periods=seasonal_periods, trend='mul',seasonal='mul')
		fit1 = model.fit(optimized=True)
		val_list_hw = fit1.forecast(len(val_list))
		temp.append(val_list_hw)

	num_of_hw = range(len(periods_list)*len(char_list)*len(para_list)**3)
	temp_key = [ ]

	for i in num_of_hw:
		variable_name = f"hw_{i}"  
		temp_key.append(variable_name)

	tempo_dic = dict(zip(temp_key,temp))
	
	return tempo_dic


# running functions
def difference(stat_dic,variable,re_dic):
    dete_ld, per_ld = diff_fun_ld(stat_dic,variable)
    dete_lw, per_lw = diff_fun_lw(stat_dic,variable)

    re_dic.update(dete_ld)
    re_dic.update(dete_lw)

# window = 1, 2, 3, 4
def moving_average(stat_dic,variable,re_dic,window,key):
    dete_ma = ha_fun(stat_dic,variable,window,key)
    re_dic.update(dete_ma)

# window = 1, 2, 3, 4
def moving_median(stat_dic,variable,re_dic,window,key):
    dete_mm = hm_fun(stat_dic,variable,window,key)
    re_dic.update(dete_mm)

def holt_winters(stat_dc,variable,re_dic):
    dete_hw = hw_fun(stat_dc,variable)
    re_dic.update(dete_hw)



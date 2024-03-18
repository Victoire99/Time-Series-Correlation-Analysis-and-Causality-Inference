'''
Author: KEWEI ZHANG
Date: 2024-02-05 14:41:22
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-05 15:42:06
FilePath: \WorkNote\Term Conclusion\classPC.py
Description: 

'''

import pandas as pd
import numpy as np
import os
import operator
from functions import *




class PC():
    def __init__(self, pc_list_num,sp_list):
        
        path = 'C:/WorkNote/datadogCorr/rawdata'
        item = os.listdir(path)    
        if not os.path.exists('C:/WorkNote/DL/output'):
            os.makedirs('C:/WorkNote/DL/output')


        pc_list = []
        for i in range(len(item)):
            if operator.contains(item[i], "xlsx") and operator.contains(item[i], "PerformanceCounter"):
                pc_list.append(item[i])

        # original input 
        self.path = path
        self.pc_list = pc_list
        self.pc_list_num = pc_list_num


        df_dic, machine_list, variables, time_list = read_data_pc(self.path, self.pc_list,self.pc_list_num)

        self.dct = df_dic
        self.variables = variables
        self.machine_list = machine_list
        self.time_list = time_list


        # dict中存储的变量写法有改变，因为加了‘Counter Instance’的后缀，所以要重新匹配选择
        # mc_list: the variables contains "MC"
        # dct_list: the variables contains "Dct"
        mc_list = [s for s in variables if "MC" in s]
        dct_list = [s for s in variables if "Dct" in s]
        # sp_list: the original spelling of the variables contains selected variables
        sp_list = sp_list + mc_list + dct_list
        # select all the selected variables
        param_list = [s for s in variables if any(sp in s for sp in sp_list)]

        sp_dict = {}
        for i in range(len(machine_list)):
            temp_dict = {}
            for j in range(len(param_list)):
                temp_dict[param_list[j]] = df_dic[machine_list[i]][param_list[j]]
                
                sp_variables = list(temp_dict.keys())

            sp_dict[machine_list[i]] = temp_dict

        #cumsum variables
        cum_list = ['Total query execution time_ ','Lock wait time (ms/sec)_ ','Total data query executions_ ']

        # 更改cum类的数据
        for i in range(len(machine_list)):
            for j in range(len(cum_list)):
                ori_array = sp_dict[machine_list[i]][cum_list[j]]
                temp_arrary = np.diff(ori_array)
                temp_arrary = np.insert(temp_arrary, 0, 0)
                sp_dict[machine_list[i]][cum_list[j]] = temp_arrary
        
        self.sp_dct = sp_dict
        self.sp_variables = sp_variables
        self.time_list = time_list
       

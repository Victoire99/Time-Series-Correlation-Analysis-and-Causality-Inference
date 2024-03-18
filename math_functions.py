'''
Author: KEWEI ZHANG
Date: 2024-01-05 14:39:02
LastEditors: KEWEI ZHANG
LastEditTime: 2024-02-12 15:20:24
FilePath: \WorkNote\Term Conclusion\math_functions.py
Description: related basic math functions

'''


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def Interpolation(data):
    data = pd.Series(data)
    data.replace(0, np.nan, inplace=True)
    mod = data.interpolate(method='linear')
    return mod


# transfer entropy visualization 
def feature_plot(dct,text,title):
    # visualization 
    print(' Computing feature importance...')

    sorted_tuples = sorted(dct.items(), key=lambda item: item[1])
    sorted_dct = dict(sorted_tuples)
    name = list(sorted_dct.keys())
    results = list(sorted_dct.values())

    plt.figure(figsize=(20,15))
    plt.barh(name, results)
    plt.yticks(range(len(name)),name)
    plt.ylim((-1,len(name)+1))
    plt.title(f'Feature Importance:{text}',size=16)
    plt.ylabel('Metrics',size=14)
    plt.xlabel(title,size=14)
    # baseline
    plt.plot([0,0],[-1,len(name)+1], '--', color='orange',
                        label=f'Baseline')
    for i in range(len(name)):
        plt.text(results[i],i,round(results[i],4),ha='left',va='center',fontsize=12)

    plt.show()
        
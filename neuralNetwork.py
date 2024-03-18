

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import layers
from tqdm import tqdm
import IPython
import IPython.display

from Window import WindowGenerator
from functions import *


# RMSE
def root_mean_squared_error(y_true, y_pred):
        
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true))) 

def nn_model():
    '''
    description: nn model,default as LSTM
    return {*}
    '''
    lstm_model = tf.keras.models.Sequential([
            # 形状 [批量数, 时间步, 特征数] => [批量数, 时间步, 单元数]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # 形状 => [批量数, 时间步, 特征数]
            tf.keras.layers.Dense(units=1)
        ])
    
    model = lstm_model
    
    return model

def compile_and_fit(model, window, patience=2):
    '''
    description: avoid overfitting
    param model {*}: nn model
    param window {*}: moving window class
    param patience {*}: 
    return {*}
    '''

    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=patience, verbose=1, mode="min", restore_best_weights=True)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[MeanAbsoluteError(), MeanSquaredError(), root_mean_squared_error, MeanAbsolutePercentageError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[lr,es])
    
    return history, model




def nn_model_vis(variables,train_df, val_df, test_df,output_path):
    '''
    description: main function for nn model and visualization
    param variables {list}: variables list
    param train_df {*}:
    param val_df {*}:
    param test_df {*}:
    param output_path {*}: 
    return {*}
    '''

    for i in range(len(variables)):
        # rnn model
        wide_window = WindowGenerator(
            input_width=24, label_width=24, shift=1,label_columns=[variables[i]],
            train_df = train_df, val_df = val_df, test_df = test_df)


        model = nn_model()
        
        print('输入形状:', wide_window.example[0].shape)
        print('输出形状:', model(wide_window.example[0]).shape) 

        history, dl_model = compile_and_fit(model, wide_window)
        IPython.display.clear_output()

        val_metrcs = model.evaluate(wide_window.val)
        metrcs = model.evaluate(wide_window.test, verbose=0)
        val_mec_dct = {'mae':val_metrcs[1],
                       'mse':val_metrcs[2],
                       'rmse':val_metrcs[3]}
        mec_dct = {'mae':metrcs[1],
                     'mse':metrcs[2],
                     'rmse':metrcs[3]}

        #*  plot 1: prediction results
        wide_window.plot(model,output_path = output_path)
        #蓝色的 Inputs 行显示每个时间步骤的输入值。模型会接收所有特征，而该绘图仅显示指定Column
        #绿色的 Labels 点显示目标预测值。这些点在预测时间，而不是输入时间显示。这就是为什么标签范围相对于输入移动了 1 步。
        #橙色的 Predictions 叉是模型针对每个输出时间步骤的预测。如果模型能够进行完美预测，则预测值将直接落在 Labels 上。


        #* plot 2L: MAE, MSE, RMSE
        # 画出预测值和真实值的对比图
        plt.figure()
        myfig = plt.gcf()
        bar_width = 0.35
        # test
        plt.bar(np.arange(len(mec_dct)), mec_dct.values(), width=bar_width, label='Test')
        # validation
        plt.bar(np.arange(len(val_mec_dct)) + bar_width, val_mec_dct.values(), width=bar_width, label='Validation')

        text = variables[i]
        text = text.replace('/', '_')
        plt.title('Evaluation Results_%s.png' % text)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.xticks(np.arange(len(mec_dct)) + bar_width / 2, mec_dct.keys())

        plt.legend()
        myfig.savefig(output_path + 'Evaluation Results_%s.png' % text )  
        #plt.show()


        #* plot 3: feature importance 横柱状图
        COLS = list(wide_window.train_df.columns)
        results = []
        print(' Computing LSTM feature importance...')

        for inputs, labels in wide_window.val.take(1):
            oof_preds = model.predict(inputs, verbose=0)
            bs_mae = np.mean(np.abs(oof_preds-labels ))
            results.append({'feature':'LSTM','mae':bs_mae})    

            X_valid = inputs.numpy()       
            y_valid = labels.numpy()

            for k in tqdm(range(len(COLS))):
                # SHUFFLE FEATURE K
                save_col = X_valid[:,:,k].copy()
                np.random.shuffle(X_valid[:,:,k])
                        
                # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
                oof_preds = model.predict(X_valid, verbose=0)
                mae = np.mean(np.abs( oof_preds-y_valid ))
                results.append({'feature':COLS[k],'mae':mae})
                X_valid[:,:,k] = save_col
                
        df_tem = pd.DataFrame(results)
        df_tem.sort_values('mae', ascending=True, inplace=True)

        plt.figure(figsize=(20,15))
        feafig = plt.gcf()
        plt.barh(np.arange(len(COLS)+1),df_tem.mae)
        plt.yticks(np.arange(len(COLS)+1),df_tem.feature.values)
        plt.title(f'Feature Importance:{variables[i]}',size=16)
        plt.ylim((-1,len(COLS)+1))
        plt.plot([bs_mae,bs_mae],[-1,len(COLS)+1], '--', color='orange',
                    label=f'Baseline OOF\nMAE={bs_mae:.3f}')
        plt.xlabel('OOF MAE with feature permuted',size=14)
        plt.ylabel('Feature',size=14)
        plt.legend()
        text = variables[i]
        text = text.replace('/', '_')
        feafig.savefig(output_path + 'Feature Importance_%s.png' % text )  

        #plt.show()
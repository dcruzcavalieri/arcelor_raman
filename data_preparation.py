# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:07:36 2023

@author: Daniel
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler


lim1 = 1; #400
lim2 = 50; #450
lim3 = 801; #1200
lim4 = 901; #1300
lim5 = 201; #600
lim6 = 401; #800
lim7 = 401; #800
lim8 = 601; #1000
lim9 = 1101; #1500
lim10 = 1301; #1800

#Sio2
model_knn_sio2 = joblib.load('model_knn_sio2.pkl')
model_svr_sio2 = joblib.load('model_svr_sio2.pkl')
model_rf_sio2 = joblib.load('model_rf_sio2.pkl')
ridge_sio2 = joblib.load('ridge_sio2.pkl')

#CaO
model_knn_cao = joblib.load('model_knn_cao.pkl')
model_svr_cao = joblib.load('model_svr_cao.pkl')
model_rf_cao = joblib.load('model_rf_cao.pkl')
ridge_cao = joblib.load('ridge_cao.pkl')

def get_data():
    data = pd.read_csv('raman_5_6_2023.csv', index_col=False)
    return data

def pipeline_transformer(data):
    
    data.index = pd.to_datetime(data["time"])
    data.drop(['time'], axis=1, inplace=True)
    
    data = data.iloc[:, :].values
    
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data.T).T
    
    return data_norm

def sinergy_vector(data):
    Xsub1 = data[0,lim1:lim2]
    Xsub2 = data[0,lim3:lim4]
    Xsub3 = data[0,lim5:lim6]
    Xsub4 = data[0,lim7:lim8]
    Xsub5 = data[0,lim9:lim10]

    data_new = np.concatenate((Xsub1,Xsub2,Xsub3,Xsub4,Xsub5),axis=0)
    data_new = data_new.reshape(1,data_new.size)
    return data_new


def sio2_cao_raman():
    data = get_data()
    
    data_new = sinergy_vector(pipeline_transformer(data))
    
    # SiO2
    y_sio2_rf_ts = model_rf_sio2.predict(data_new)
    y_sio2_knn_ts = model_knn_sio2.predict(data_new)
    y_sio2_svr_ts = model_svr_sio2.predict(data_new)
    y_sio2_ridge_ts = ridge_sio2.predict(data_new)

    # CaO
    y_cao_rf_ts = model_rf_cao.predict(data_new)
    y_cao_knn_ts = model_knn_cao.predict(data_new)
    y_cao_svr_ts = model_svr_cao.predict(data_new)
    y_cao_ridge_ts = ridge_cao.predict(data_new)

    # B2
    X_ts_b2 = np.vstack((y_cao_knn_ts, y_cao_svr_ts, y_cao_rf_ts, 
                         y_cao_ridge_ts, y_sio2_knn_ts, y_sio2_svr_ts, 
                         y_sio2_rf_ts, y_sio2_ridge_ts)).T
    
    return X_ts_b2
    
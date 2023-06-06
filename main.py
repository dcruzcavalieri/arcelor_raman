# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:32:10 2023

@author: Daniel
"""

import joblib
from flask import Flask, request, jsonify
from data_preparation import sio2_cao_raman
from data_preparation import get_data

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def predict():
    #data = request.get_json()
    #print(data)
    X_b2_ts = sio2_cao_raman()
    clf = joblib.load('model_knn_b2.pkl')
    prediction = clf.predict(X_b2_ts)
    print(prediction)
    result = {
        'b2_prediction': list(prediction)
    }
    return jsonify(result)

if __name__ == '__main__':
     app.run(port=9696)

#%%

import requests

url = "http://localhost:9696/predict"
r = requests.post(url)
r.text.strip()


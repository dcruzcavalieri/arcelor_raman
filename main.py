# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:32:10 2023

@author: Daniel

"""

import pandas as pd
import joblib
from flask import Flask, session, render_template
from data_preparation import sio2_cao_raman
import csv
import numpy as np
 
UPLOAD_FOLDER = 'D:/Raman/Fuzzy/model_files/uploads/'
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__)
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'Foi'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # upload file flask
    data_filename = 'D:/Raman/Fuzzy/model_files/uploads/raman_5_6_2023.csv'
    session['uploaded_data_file_path'] = data_filename
    
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    
    # read csv
    data = pd.read_csv(data_file_path, index_col=False)
    data_raman = sio2_cao_raman(data)
    
    cao = np.mean(data_raman[0,:3])
    sio2 = np.mean(data_raman[0,3:])
    
    clf = joblib.load('D:/Raman/Fuzzy/model_files/models/model_knn_b2.pkl')
    b2 = clf.predict(data_raman)
    
    #print(b2[0])
    #print(sio2)
    #print(cao)
    
    result = [
        {'b2': b2[0]}, 
        {'SiO2': sio2},
        {'CaO': cao},
    ]
    
    # write to file
    output_file = 'D:/Raman/Fuzzy/model_files/results/raman_predictions.csv'
    with open(output_file, "a", newline='') as f:
      # Create the output csv file
      csv_writer = csv.DictWriter(f, fieldnames=["b2","SiO2","CaO"])
      # Header: b2
      csv_writer.writerow({"b2": b2[0], "SiO2": sio2, "CaO": cao})
    
    return render_template('index.html', prediction_text=result)
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    


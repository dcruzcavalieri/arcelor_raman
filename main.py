# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:32:10 2023

@author: Daniel


import joblib
import pandas as pd
from flask import Flask, jsonify
from data_preparation import sio2_cao_raman

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def predict():
    data = pd.read_csv('./uploads/raman_5_6_2023.csv', index_col=False)
    data_raman = sio2_cao_raman(data)
    
    clf = joblib.load('./models/model_knn_b2.pkl')
    prediction = clf.predict(data_raman)
    print(prediction)
    
    result = {
        'b2_prediction': list(prediction)
    }
    return jsonify(result)

if __name__ == '__main__':
     app.run(host='0.0.0.0')

"""

import pandas as pd
import joblib
from flask import Flask, jsonify, session
from data_preparation import sio2_cao_raman
import csv
 
UPLOAD_FOLDER = 'D:/Raman/Fuzzy/model_files/uploads/'
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__)
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'Foi'
 
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
    
    clf = joblib.load('./models/model_knn_b2.pkl')
    prediction = clf.predict(data_raman)
    print(prediction)
    
    result = {
        'b2_prediction': list(prediction)
    }
    
    # write to file
    output_file = 'D:/Raman/Fuzzy/model_files/results/raman_predictions.csv'
    with open(output_file, "a", newline='') as f:
      # Create the output csv file
      csv_writer = csv.DictWriter(f, fieldnames=["b2"])
      #csv_writer.writeheader()
      csv_writer.writerow({"b2": prediction})
    
    return jsonify(result)
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    


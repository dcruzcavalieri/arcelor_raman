#Using the base image with python 3.7
FROM python:3.8
 
#Set our working directory as app
WORKDIR /model_files

#Installing python packages pandas, scikit-learn and gunicorn
RUN pip install pandas scikit-learn flask gunicorn
 
# Copy the models directory and server.py files
ADD ./models ./models
ADD main.py main.py
 
 #Exposing the port 5000 from the container
 EXPOSE 5000
 #Starting the python application
 CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
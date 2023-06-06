#Using the base image with python 3.7
FROM python:3.8.16 

#Set our working directory as app
WORKDIR /model_files

#Installing python packages pandas, scikit-learn and gunicorn
RUN pip install pandas scikit-learn flask numpy joblib
 
COPY . /model_files

EXPOSE 5000
 
CMD ["python3", "/model_files/main.py", "--host", "0.0.0.0"]
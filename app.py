# Library Imports for FastApi
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from IncomeClassifier import IncomeClassifier

#from flask import Flask,render_template,request,jsonify
#from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import pickle
#from xgboost import xg
import sklearn

# Creating the application object
app = FastAPI()
# HTML template folder
#templates = Jinja2Templates(directory='templates')
# Loading the model file
model = pickle.load(open('income_classifier_xgb.pickle','rb'))

@app.get('/') # route to display home page
#@cross_origin()
def home():
    #return render_template('index.html')
    return {'message' : "Hello World"}

@app.post('/predict') # route to display prediction page
#@cross_origin()
def predict(data:IncomeClassifier):
    data = data.dict()
    age = data['age']
    occupation = data['occupation']
    if occupation == 'Manager':
        occupation = 4066
    elif occupation == 'Specialist':
        occupation = 5983
    elif occupation == 'Sales':
        occupation = 3650
    elif occupation == 'Clerical':
        occupation = 3770
    elif occupation == 'Cleaners':
        occupation = 1370
    elif occupation == 'Other-service':
        occupation = 3295
    elif occupation == 'Craft-repair':
        occupation = 4099
    sex = data['sex']
    if sex == 'Male':
        sex = 0
    else:
        sex = 1
    capital_gain = data['capital_gain']
    capital_loss = data['capital_loss']
    hours_per_week = data['hours_per_week']
    workclass = data['workclass']
    if workclass == 'Govt_emp':
        workclass_Others = 0
        workclass_Private = 0
        workclass_Self_emp = 0
    elif workclass == 'Private':
        workclass_Others = 0
        workclass_Private = 1
        workclass_Self_emp = 0
    elif workclass == 'Self_emp':
        workclass_Others = 0
        workclass_Private = 0
        workclass_Self_emp = 1
    else:
        workclass_Others = 1
        workclass_Private = 0
        workclass_Self_emp = 0
    education = data['education']
    if education == 'High':
        education_Low = 0
        education_Medium = 0
    elif education == 'Medium':
        education_Low = 0
        education_Medium = 1
    else:
        education_Low = 1
        education_Medium = 0
    marital_status = data['marital_status']
    if marital_status == 'Married':
        marital_status_Married = 1
        marital_status_Separated = 0
        marital_status_Single = 0
        marital_status_Widowed = 0
    elif marital_status == 'Separated':
        marital_status_Married = 0
        marital_status_Separated = 1
        marital_status_Single = 0
        marital_status_Widowed = 0
    elif marital_status == 'Single':
        marital_status_Married = 0
        marital_status_Separated = 0
        marital_status_Single = 1
        marital_status_Widowed = 0
    elif marital_status == 'Widowed':
        marital_status_Married = 0
        marital_status_Separated = 0
        marital_status_Single = 0
        marital_status_Widowed = 1
    else:
        marital_status_Married = 0
        marital_status_Separated = 0
        marital_status_Single = 0
        marital_status_Widowed = 0
    race = data['race']
    if race == 'Black':
        race_Other = 0
        race_White = 0
    elif race == 'White':
        race_Other = 0
        race_White = 1
    else:
        race_Other = 1
        race_White = 0
    native_country = data['native_country']
    if native_country == 'Germany':
        native_country_Germany = 1
        native_country_Mexico = 0
        native_country_Others = 0
        native_country_Philippines = 0
        native_country_United_States = 0
    elif native_country == 'Mexico':
        native_country_Germany = 0
        native_country_Mexico = 1
        native_country_Others = 0
        native_country_Philippines = 0
        native_country_United_States = 0
    elif native_country == 'Philippines':
        native_country_Germany = 0
        native_country_Mexico = 0
        native_country_Others = 0
        native_country_Philippines = 1
        native_country_United_States = 0
    elif native_country == 'United_States':
        native_country_Germany = 0
        native_country_Mexico = 0
        native_country_Others = 0
        native_country_Philippines = 0
        native_country_United_States = 1
    elif native_country == 'Canada':
        native_country_Germany = 0
        native_country_Mexico = 0
        native_country_Others = 0
        native_country_Philippines = 0
        native_country_United_States = 0
    else:
        native_country_Germany = 0
        native_country_Mexico = 0
        native_country_Others = 1
        native_country_Philippines = 0
        native_country_United_States = 0

    #Creating a dictionary with the new variables before passing to the model
    dict_pred = {'age': age, 'occupation': occupation,'sex': sex ,'capital_gain': capital_gain, 'capital_loss': capital_loss,
               'hours_per_week': hours_per_week, 'workclass_Others': workclass_Others,
               'workclass_Private': workclass_Private,'workclass_Self_emp': workclass_Self_emp,
               'education_Low':education_Low,'education_Medium':education_Medium,
               'marital_status_Married':marital_status_Married,'marital_status_Separated':marital_status_Separated,
               'marital_status_Single':marital_status_Single,'marital_status_Widowed':marital_status_Widowed,
               'race_Other':race_Other,'race_White':race_White,'native_country_Germany':native_country_Germany,
               'native_country_Mexico':native_country_Mexico, 'native_country_Others': native_country_Others,
               'native_country_Philippines':native_country_Philippines,
               'native_country_United_States':native_country_United_States}

    data_df = pd.DataFrame(dict_pred, index=[0, ])
    prediction = model.predict(data_df)

    #prediction = model.predict([[]])

    if (prediction[0]>0.5):
        prediction = 'Yearly Income is above 50K'
    else:
        prediction = 'Yearly Income is less than or equal to 50K'

    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port = 8000)

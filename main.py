from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import pickle
# from xgboost import xg
import sklearn

# Creating Flask application object
app = Flask(__name__)
# Loading the model file
model = pickle.load(open('income_classifier_xgb.pickle', 'rb'))

@app.route('/', methods = ['GET'])  # route to display home page
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST']) #Prediction API
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])

            occupation = request.form['occupation']
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

            sex = int(request.form['sex'])

            capital_gain = int(request.form['capital_gain'])

            capital_loss = int(request.form['capital_loss'])

            hours_per_week = int(request.form['hours_per_week'])

            workclass = request.form['workclass']
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
            education = request.form['education']
            if education == 'High':
                education_Low = 0
                education_Medium = 0
            elif education == 'Medium':
                education_Low = 0
                education_Medium = 1
            else:
                education_Low = 1
                education_Medium = 0
            marital_status = request.form['marital_status']
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
            race = request.form['race']
            if race == 'Black':
                race_Other = 0
                race_White = 0
            elif race == 'White':
                race_Other = 0
                race_White = 1
            else:
                race_Other = 1
                race_White = 0
            native_country = request.form['native_country']
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

            # Creating a dictionary with the new variables before passing to the model
            dict_pred = {'age': age, 'occupation': occupation, 'sex': sex, 'capital_gain': capital_gain,
                         'capital_loss': capital_loss,
                         'hours_per_week': hours_per_week, 'workclass_Others': workclass_Others,
                         'workclass_Private': workclass_Private, 'workclass_Self_emp': workclass_Self_emp,
                         'education_Low': education_Low, 'education_Medium': education_Medium,
                         'marital_status_Married': marital_status_Married,
                         'marital_status_Separated': marital_status_Separated,
                         'marital_status_Single': marital_status_Single,
                         'marital_status_Widowed': marital_status_Widowed,
                         'race_Other': race_Other, 'race_White': race_White,
                         'native_country_Germany': native_country_Germany,
                         'native_country_Mexico': native_country_Mexico, 'native_country_Others': native_country_Others,
                         'native_country_Philippines': native_country_Philippines,
                         'native_country_United_States': native_country_United_States}

            data_df = pd.DataFrame(dict_pred, index=[0, ])
            # print(data_df)
            prediction = model.predict(data_df)
            # printing the prediction value
            #print(prediction)
            # prediction = model.predict([[]])

            if (prediction[0] > 0.5):
                prediction = 'Yearly Income is above 50K'
            else:
                prediction = 'Yearly Income is less than or equal to 50K'

            return render_template('results.html', prediction=prediction)


        except Exception as e:
            print("The Exception message is:", e)
            return jsonify("error:Something is wrong")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(host='0.0.0.0', port=8001,debug=True) # running the app

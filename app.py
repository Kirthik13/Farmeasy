# Importing essential libraries and modules

from tkinter.messagebox import NO
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzytree import FuzzyDecisionTreeClassifier
from flask_fontawesome import FontAwesome
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# from torchvision import transforms

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------



# Loading crop recommendation model

# crop_recommendation_model_path = 'models/FuzzyDecisionTree.pkl'
# crop_recommendation_model = pickle.load(
#     open(crop_recommendation_model_path, 'rb'))

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Loading crop yield model by state

crop_yield_state_model_path = 'models/DecisionTree_State.pkl'
crop_yield_state_model = pickle.load(
    open(crop_yield_state_model_path, 'rb'))

# Loading crop yield model by district

crop_yield_dist_model_path = 'models/DecisionTree_Dist.pkl'
crop_yield_dist_model = pickle.load(
    open(crop_yield_dist_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None




# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
fa = FontAwesome(app)

# render home page



@ app.route('/')
def home():
    title = 'FarmEasy-Home'
    return render_template('index.html', title=title)



# render crop recommendation form page


# @ app.route('/crop-recommend')
# def crop_recommend():
#     title = 'Harvestify - Crop Recommendation'
#     return render_template('index.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)
@ app.route('/cropyield')
def yield_predict():
    title = 'Harvestify - Yield Prediction'

    return render_template('cropyield.html', title=title)

@ app.route('/crop-result')
def crop_res():
    title = 'Harvestify - Crop Recommendation'
   

    return render_template('cropresult.html', title=title)




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'
    global name_of_crop
    if request.method == 'POST':
        print("asdihasd")
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        # print(N)
        # print(P)
        # print(K)
        # print(ph)
        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            name_of_crop=final_prediction
            return render_template('cropresult.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)
    
# crop yield
@ app.route('/yield-predict', methods=['POST'])
def crop_yield_prediction():
    title = 'Harvestify - Crop Yield Prediction'
    
    if request.method == 'POST':
        print("asdihasd")
        # N = int(request.form['nitrogen'])
        # P = int(request.form['phosphorous'])
        # K = int(request.form['potassium'])
        area = float(request.form['area'])
        # rainfall = float(request.form['rainfall'])
        # print(N)
        # print(P)
        # print(K)
        # print(ph)
        season = request.form.get("season")
        state = request.form.get("stt")
        city = request.form.get("city")

        # if weather_fetch(city) != None:
            # temperature, humidity = weather_fetch(city)
        # data = [state,season,name_of_crop]

        df = pd.read_csv('Data/crop_production.csv')
        data = df.dropna()
        test = df[~df["Production"].notna()].drop("Production",axis=1)
        data['yield']=data['Production']/data['Area']
        data1 = data.drop(["District_Name","Crop_Year","Area","Production"],axis=1)
        data_dum = pd.get_dummies(data1)
        x = data_dum.drop("yield",axis=1)
        y = data_dum[["yield"]]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
        regressor = DecisionTreeRegressor(random_state=42)
        regressor.fit(x_train,y_train)

        predict = pd.DataFrame({'State_Name':[state],
                           'Season': [season],
                        'Crop': [name_of_crop]})
 
        predict = pd.get_dummies(predict)
        predict = predict.reindex(columns=x_train.columns,fill_value=0)
        # my_prediction =crop_yield_state_model.predict(predict)
        my_prediction=regressor.predict(predict)
        a = my_prediction[0]*area
        a = int((a * 100) + 0.5) / 100.0
        final_prediction=a
        # format(final_prediction, '.2f')

            # print(state)
            # print(city)
        return render_template('yield-result.html', recommendation=final_prediction, title=title)

        # else:

        #     return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    print(name_of_crop)
    crop_name = str(name_of_crop)
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['potassium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)





# ===============================================================================================
if __name__ == '__main__':
    app.debug = True
    
    app.run()
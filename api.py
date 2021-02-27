from flask import Flask, jsonify, request
import psycopg2
import os, subprocess
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
from functools import reduce
import operator

app = Flask(__name__)


""" Endpoint for testing
""" 

"""Testing """   
@app.route('/testing', methods = ['GET'])
def get_prediction():
    #make_response(jsonify({"message": "Testing Api"}), 200)
    json = {
        "message": "Testing Api"
        }
    
    return json


""" Endpoint that receive an input from the user
"""

@app.route('/API/PREDICT_AVG_SENTIMENT', methods=['POST'])
def predict_avg_sentiment():
  
    hos = request.json['hold out samples']
    print(hos)
    lo = request.json['lag observations']
    print(lo)
    dd = request.json['degree of differencing']
    print(dd)
    maw = request.json['moving average window']
    print(maw)
    
    conn = psycopg2.connect(host="ec2-52-63-12-173.ap-southeast-2.compute.amazonaws.com", port = 5432, database="misc", user="hiring_test_readonly", password="pretense_yarrow_armhole")
    
    sql_query = "SELECT publish_datetime, date_time_utc, date_time_aest, sentiment_score, tags, sector, category, author, article_title, article_summary, sub_category, article_content FROM afr_articles ORDER BY publish_datetime"
    
    table = pd.read_sql_query(sql_query, conn)
    
    table['publish_date'] = [datetime.datetime.date(d) for d in table['publish_datetime']] 

    sentiment_by_day = table.pivot_table(index='publish_date', values='sentiment_score', aggfunc='mean')

    X = sentiment_by_day.values
     
    size = len(X) - hos
    
    train, test = X[0:size], X[size:]
    
    history = [x for x in train]
    
    predictions = list()
    
    response = {"Predictions": list()}
    
    for t in range(len(test)):
        model = ARIMA(history, order=(lo,dd,maw))
        model_fit = model.fit()
        ar_coef = model_fit.arparams
        yhat = predict(ar_coef, history)
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
        response["Predictions"].append(['predicted:' + str(yhat), 'expected:' + str(obs)])
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    
    next_day = model_fit.predict(start=len(train)+len(test), end=len(train)+len(test), dynamic=False)
    print(next_day)
    response["Predictions"].append(['average sentiment of the next day.:' , str(next_day)])
    
    return jsonify(response) 

def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef)+1):
        yhat += coef[i-1] * history[-i]
    return yhat

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)  # Running the application in the port 9090
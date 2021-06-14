import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
import azureml.automl.core
from flask import Flask, request


time_forecasting_model = joblib.load("./time_forecasting.pkl")
regresstion_model = joblib.load("./regression.pkl")
crop_recomentation_model = joblib.load("./crop_recomentation.pkl")

app = Flask(__name__)


def time_forecast_temp(data):
    try:
        y_query = None
        if 'y_query' in data.columns:
            y_query = data.pop('y_query').values
        result = time_forecasting_model.forecast(data, y_query)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

    forecast_as_list = result[0].tolist()
    index_as_df = result[1].index.to_frame().reset_index(drop=True)

    return json.dumps({"forecast": forecast_as_list,   # return the minimum over the wire:
                       # no forecast and its featurized values
                       "index": json.loads(index_as_df.to_json(orient='records'))
                       })


def regression_temp(data):
    try:
        result = regresstion_model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})


def crop_recomentation(data):
    try:
        result = crop_recomentation_model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})


@app.route('/ai/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        input_datetime = pd.DataFrame({"Datetime": pd.Series(
            [req_data['Datetime']], dtype="datetime64[ns]")})
        input_general = pd.DataFrame({"temperature": pd.Series([req_data['temperature']]), "humidity": pd.Series(
            [req_data['humidity']]), "ph": pd.Series([req_data['ph']])})
        print("datetime: ")
        print(input_datetime)
        print("general: ")
        print(input_general)
        time_forecast_result = json.loads(time_forecast_temp(input_datetime))
        regression_result = json.loads(regression_temp(input_datetime))
        crop_recomentation_result = json.loads(crop_recomentation(input_general))
        return json.dumps({"date": req_data['Datetime'],"crop_recomentation": crop_recomentation_result["result"][0], "time_forecast": time_forecast_result["forecast"][0], "regresstion": regression_result["result"][0]})
    except Exception as e:
        return json.dumps(str(e))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=True)

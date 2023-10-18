from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            bid_opening=float(request.form.get('BO')),
            bid_highest=float(request.form.get('BH')),
            bid_lowest=float(request.form.get('BL')),
            bid_change=float(request.form.get('BCh')),
            ask_open=float(request.form.get('AO')),
            ask_highest=float(request.form.get('AH')),
            ask_lowest=float(request.form.get('AL')),
            ask_close=float(request.form.get('AC')),
            ask_change=float(request.form.get('ACh'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.debug = True
    app.run(host="0.0.0.0")
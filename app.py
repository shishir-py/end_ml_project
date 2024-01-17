from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

import pandas as pd
#init app
app= Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict_data', methods=['GET','POST'] )
def predict_data():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score')))
        df=data.get_data_as_df()
        print("before prediction")
        
        predict_pipeline=PredictionPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)

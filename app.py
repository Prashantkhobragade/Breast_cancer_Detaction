from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from BreastCancer.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

#Route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            radius_mean = request.get('radius_mean'),
            texture_mean = request.get('texture_mean'),
            perimeter_mean = request.get('perimeter_mean'),
            area_mean = request.get('area_mean'),
            smoothness_mean = request.get('smoothness_mean'),
            compactness_mean = request.get('compactness_mean'),
            concavity_mean = request.get('concavity_mean'),
            concave_points_mean = request.get('concave_points_mean'),
            symmetry_mean = request.get('symmetry_mean'),
            fractal_dimension_mean = request.get('fractal_dimension_mean'),
            radius_se = request.get('radius_se'),
            texture_se = request.get('texture_se'),
            perimeter_se = request.get('perimeter_se'), 
            area_se = request.get('area_se'),
            smoothness_se = request.get('smoothness_se'),
            compactness_se = request.get('compactness_se'),
            concavity_se = request.get('concavity_se'),
            concave_points_se = request.get('concave_points_se'),
            symmetry_se = request.get('symmetry_se'),
            fractal_dimension_se = request.get('fractal_dimension_se'),
            radius_worst = request.get('radius_worst'),
            texture_worst = request.get('texture_worst'),
            perimeter_worst = request.get('perimeter_worst'),
            area_worst = request.get('area_worst'),
            smoothness_worst = request.get('smoothness_worst'),
            compactness_worst = request.get('compactness_worst'),
            concavity_worst = request.get('concavity_worst'),
            concave_points_worst = request.get('concave_points_worst'),
            symmetry_worst = request.get('symmetry_worst'),
            fractal_dimension_worst= request.get('fractal_dimension_worst')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)


        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True) 
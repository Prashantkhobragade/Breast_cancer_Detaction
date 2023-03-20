import sys
import os
import pandas as pd
import numpy as np
from BreastCancer.exception import CustomException
from BreastCancer.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(feature)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,
            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst):
        
        self.radius_mean = radius_mean
        self.texture_mean = texture_mean
        self.perimeter_mean = perimeter_mean
        self.area_mean = area_mean
        self.smoothness_mean = smoothness_mean
        self.compactness_mean = compactness_mean
        self.concave_points_mean = concave_points_mean
        self.symmetry_mean = symmetry_mean
        self.fractal_dimension_mean = fractal_dimension_mean
        self.radius_worst = radius_worst
        self.texture_worst = texture_worst
        self.perimeter_worst = perimeter_worst
        self.area_worst = area_worst
        self.smoothness_worst = smoothness_worst
        self.compactness_worst = compactness_worst
        self.concavity_worst = concavity_worst
        self.concave_points_worst = concave_points_worst
        self.symmetry_worst = symmetry_worst
        self.fractal_dimension_worst = fractal_dimension_worst

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "radius_mean": [self.radius_mean],
                "texture_mean": [self.texture_mean],
                "perimeter_mean": [self.perimeter_mean],
                "area_mean": [self.area_mean],
                "smoothness_mean": [self.smoothness_mean],
                "compactness_mean" : [self.compactness_mean],
                "concave_points_mean": [self.concave_points_mean],
                "symmetry_mean": [self.symmetry_mean],
                "fractal_dimension_mean": [self.fractal_dimension_mean],
                "radius_worst": [self.radius_worst],
                "texture_worst": [self.texture_worst],
                "perimeter_worst" : [self.perimeter_worst],
                "area_worst" : [self.area_worst],
                "smoothness_worst" : [self.smoothness_worst],
                "compactness_worst" : [self.compactness_worst],
                "concavity_worst" : [self.concavity_worst],
                "concave_points_worst" : [self.concave_points_worst],
                "symmetry_worst" : [self.symmetry_worst],
                "fractal_dimension_worst" : [self.fractal_dimension_worst]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)


            
                
        
        
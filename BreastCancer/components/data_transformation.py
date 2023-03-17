import sys
from dataclasses import dataclass
from BreastCancer.exception import CustomException
from BreastCancer.logger import logging
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from BreastCancer.utils import save_object 


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationconfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns = [
                'radius_mean','texture_mean',
                'perimeter_mean','area_mean',
                'smoothness_mean','compactness_mean',
                'concavity_mean', 'concave points_mean','symmetry_mean',
                'fractal_dimension_mean','radius_se','texture_se',
                'perimeter_se','area_se','smoothness_se',
                'compactness_se','concavity_se','concave points_se',
                'symmetry_se','fractal_dimension_se','radius_worst',
                'texture_worst','perimeter_worst','area_worst',
                'smoothness_worst','compactness_worst','concavity_worst',
                'concave points_worst','symmetry_worst','fractal_dimension_worst'
                ]
            
            categorical_columns =['diagnosis']

            num_pipline = Pipeline(
                steps=[
                ("scaler",StandardScaler())
                ]
            )

            cat_pipline = Pipeline(
                steps=[
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Categorical columns : {categorical_columns}")

            logging.info(f"Numerical columns : {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipline",num_pipline,numerical_columns),
                ("cat_pipline",cat_pipline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read training and testing data")

            logging.info("Obtaining preprocessor object")

            preprocesssing_obj = self.get_data_transformer_object()

            target_column_name = "diagnosis"
            numerical_columns = [
                'radius_mean','texture_mean',
                'perimeter_mean','area_mean',
                'smoothness_mean','compactness_mean',
                'concavity_mean', 'concave points_mean','symmetry_mean',
                'fractal_dimension_mean','radius_se','texture_se',
                'perimeter_se','area_se','smoothness_se',
                'compactness_se','concavity_se','concave points_se',
                'symmetry_se','fractal_dimension_se','radius_worst',
                'texture_worst','perimeter_worst','area_worst',
                'smoothness_worst','compactness_worst','concavity_worst',
                'concave points_worst','symmetry_worst','fractal_dimension_worst'
                ]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Appling preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocesssing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesssing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocesssing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
            
            

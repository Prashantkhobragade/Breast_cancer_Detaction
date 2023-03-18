import os
import sys 
from dataclasses import dataclass

import pandas as pd
import numpy as np


from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from BreastCancer.exception import CustomException
from BreastCancer.logger import logging
from BreastCancer.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Spliting training and test data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest' : RandomForestClassifier(),
                'Decision Tree' : DecisionTreeClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'K-Neighbors Classifier': KNeighborsClassifier(),
                'XGB Classifier': XGBClassifier(),
                'Cat Boost Classifier': CatBoostClassifier(verbose=False),
                'Ada Boost Classifier': AdaBoostClassifier()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test,
                                               models = models)
            
            ## to get best model score from the dict
            best_model_score = max(sorted(model_report.values()))

            #to get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No best model found")
            logging.info("Best model found on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            
            
        except Exception as e:
            raise CustomException(e,sys)



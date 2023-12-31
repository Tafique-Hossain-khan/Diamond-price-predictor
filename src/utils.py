import os
from src.exception import CustomException
from src.logger import logging
import sys
import numpy as np
import pandas as pd
import pickle 

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj,f)


    except Exception as e:
        raise CustomException(e,sys)
    
#code for module for evaluation
def evaluate_model(X_train,y_train,X_test,y_test,models):
    
    try:
        report ={}
        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            # predict Testing data
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report


    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)
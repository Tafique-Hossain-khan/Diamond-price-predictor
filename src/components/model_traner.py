from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from dataclasses import dataclass
import os,sys

from src.utils import evaluate_model

@dataclass
class ModelTranerConfig:
    model_file_path:str = os.path.join("artifacts","model.pkl")

class ModelTraner:
    def __init__(self) -> None:
        self.model_traner_config = ModelTranerConfig()

    def initiate_model_traner(self,train_arr,test_arr):

        try:
            logging.info('Spliting the train and test array')
            '''X_train, y_train, X_test, y_test = (
            train_arr[:,:-1],
            train_arr[:-1],
            test_arr[:,:-1],
            test_arr[:-1]
            )'''
            features = train_arr[:, :-1]
            target = train_arr[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30,random_state=42)


            
            models = { 
            'Linear Regression': LinearRegression(),
            'Lasso':Lasso(),
            'Ridge Regression': Ridge(),
            'Descision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
            }   

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print("\n====================================================================================")
            logging.info(f'Model Report : {model_report}')

            # To get the best model report

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            berst_model = models[best_model_name]

            print(f"Best model found, Model Name:{best_model_name}, R2 Score : {best_model_score}")
            print("\n====================================================================================")
            logging.info(f"Best model found, Model Name:{best_model_name}, R2 Score : {best_model_score}")

            save_object(
                file_path=ModelTranerConfig.model_file_path,
                obj=berst_model
            )

        except Exception as e:
            raise CustomException(e,sys)
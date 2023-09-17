from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os,sys
import numpy as np
import pandas as pd

from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transofmation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):

        try:
            logging.info('Data transformation initiated')

            num_col = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_col = ['cut', 'color', 'clarity']

            #Defining the ranking for catogerical variable
            cut_category = ['Fair','Good','Very Good','Ideal','Premium']
            color_category = ['D','E','F','G','H','I','J']
            clarity_category = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Data Transition is staring')
            ## pipeline for numerical col

            pipe_num = Pipeline([
                ('imputer',SimpleImputer(missing_values=np.nan,strategy='mean')),
                ('stander_scaler',StandardScaler())
            ])

            #categorial col

            pipe_cat = Pipeline(steps=[
                ('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[cut_category,color_category,clarity_category])),

            ])

            preprocessor = ColumnTransformer([
                ('num_trf',pipe_num,num_col),
                ('cat_trf',pipe_cat,cat_col)
            ])
            logging.info('Data Transformation complited')
            return preprocessor



        except Exception as e:
            raise CustomException(e,sys)

    # performing the data transformation with the data

    def initiate_data_transformation(self,train_data,test_data):

        try:
            logging.info('Initiating the data transformation')
            # need to get the data
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            # getting the pre processor obj
            preprocessor_obj = self.get_data_transformation_obj()

            #crating the data (X_train ...) for the transformaton
            
            target_col = ['price']
            drop_col = ['price','id']

            #for train data
            input_feature_train_df = train_df.drop(drop_col,axis=1)
            target_feature_train_df = train_df[target_col]

            #for test data
            input_feature_test_df = test_df.drop(drop_col,axis=1)
            target_feature_test_df = test_df[target_col]

            #data transformation
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(file_path=self.data_transofmation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)
            
            return(
                train_arr,
                test_arr,
                self.data_transofmation_config.preprocessor_obj_file_path
            )
            



        except Exception as e:
            raise CustomException(e,sys)


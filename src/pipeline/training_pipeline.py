
from src.components.data_transfromation import DataTransformation
from src.components.data_injection import DataInjection
from src.logger import logging
from src.components.model_traner import ModelTraner


if __name__ == "__main__":
    
    obj = DataInjection()
    train_data,test_data=obj.initiate_data_ingestion()
    print(train_data,test_data)
    preprocessor_obj = DataTransformation()
    train_arr,test_arr,obj_path = preprocessor_obj.initiate_data_transformation(train_data=train_data,test_data=test_data)


    model_traner = ModelTraner()
    model_traner.initiate_model_traner(train_arr,test_arr)
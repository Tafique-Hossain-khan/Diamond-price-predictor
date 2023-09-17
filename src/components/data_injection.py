import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transfromation import DataTransformation

##any input we will write in this class
## Making this class as dataclass we dont need constructor to initiliize the variable
@dataclass
class DataInjectionConfig:
    # This class will contain the path for all the data
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','rawdata.csv')


class DataInjection:
    def __init__(self) -> None:
        self.injection_config = DataInjectionConfig()
       
    # Now steps for reading the data
    def initiate_data_ingestion(self):
        logging.info('The data ingestion method')
        try:
            
            df = pd.read_csv('notebook\data\gemstone.csv')
            logging.info('Read the dataset as dataframe')
            
            #os.makedirs(os.path.dirname(self.injection_config.train_data_path),exist_ok=True)
            #os.makedirs(os.path.dirname(self.injection_config.test_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.injection_config.raw_data_path),exist_ok=True)

            df.to_csv(self.injection_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated')

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.injection_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.injection_config.test_data_path,index=False,header=True)
            logging.info('Data injection complicated')

            return(
                self.injection_config.train_data_path,
                self.injection_config.test_data_path,
                
            )

        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataInjection()
    train_data,test_data=obj.initiate_data_ingestion()
    
    preprocessor_obj = DataTransformation()
    preprocessor_obj.initiate_data_transformation(train_data=train_data,test_data=test_data)
import os,sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from src.utils import load_object

class PredictPipeline:

    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            logging.info(f'The price is{pred}')
            return pred

        except Exception as e:
            raise CustomException(e,sys)

#class for the inpute data

class CustomInput:
    def __init__(self,carat:float,cut:str,color:str,clarity:str,depth:float ,table:float ,x:float,y:float,z:float):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth= depth
        self.table= table
        self.x = x
        self.y= y
        self.z = z

    def get_df(self):
        logging.info("Crating the dataframe")
        try:
            custom_data_input = {

            "carat":[self.carat],
            'cut': [self.cut],
            'color': [self.color],
            'clarity':[self.clarity],
            'depth':[self.depth],
            'table':[self.table],
            'x':[self.x],
            'y':[self.y],
            'z':[self.z]
            }
            logging.info(pd.DataFrame(custom_data_input))
            
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e,sys)
        

'''if __name__ == "__main__":
     
    obj = CustomInput(0.32,'Ideal',	'G','VS1',	61.6,	56.0,	4.38,4.41 ,2.71)
    featers = obj.get_df()
    predict_obj = PredictPipeline()
    predict_obj.predict(featers)'''
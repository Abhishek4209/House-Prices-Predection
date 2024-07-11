from src.logger import logging
from src.exception import CustomException
from src.utils import load_objects
import os 
import sys
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass
    
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            
            preprocessor=load_objects(preprocessor_path)
            model=load_objects(model_path)
            
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            
            return pred

        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Some Exception Occured in Predict function in Prediction Pipeline Function')
            

class CustomData:
    def __init__(self,
                NoOfBedrooms:float,
                NoOfBathrooms:float,
                NoOfFloors:float,
                FlatArea:float,
                LotArea:float,
                BasementArea:float,
                AreaOfTheHouseFromBasement:float,
                LivingAreaAfterRenovation:float,
                LotAreaAfterRenovation:float,
                AgeOfHouse:float,
                ConditionOfTheHouse:str,
                OverallGrade:float):

        self.NoOfBedrooms=NoOfBedrooms
        self.NoOfBathrooms=NoOfBathrooms
        self.NoOfFloors=NoOfFloors
        self.FlatArea=FlatArea
        self.LotArea=LotArea
        self.BasementArea=BasementArea
        self.AreaOfTheHouseFromBasement=AreaOfTheHouseFromBasement
        self.LivingAreaAfterRenovation=LivingAreaAfterRenovation
        self.LotAreaAfterRenovation=LotAreaAfterRenovation
        self.AgeOfHouse=AgeOfHouse
        self.ConditionOfTheHouse=ConditionOfTheHouse
        self.OverallGrade=OverallGrade
    
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'NoOfBedrooms':[self.NoOfBedrooms],
                'NoOfBathrooms':[self.NoOfBathrooms],
                'NoOfFloors':[ self.NoOfFloors],
                'FlatArea':[ self.FlatArea],
                'LotArea':[self.LotArea],
                'BasementArea':[self.BasementArea],
                'AreaOfTheHouseFromBasement':[self.AreaOfTheHouseFromBasement],
                'LivingAreaAfterRenovation':[self.LivingAreaAfterRenovation],
                'LotAreaAfterRenovation':[self.LotAreaAfterRenovation],
                'AgeOfHouse':[self.AgeOfHouse],
                'ConditionOfTheHouse':[self.ConditionOfTheHouse],
                'OverallGrade':[self.OverallGrade]
            }
            
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            # logging.info(df)
            return df 
        
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Exception Occured in Predection Pipeline')
            
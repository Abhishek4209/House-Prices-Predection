import numpy as np   
import pandas as pd
import os 
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

### Data ingestion Config class: 
@dataclass
class DataIngestionConfig():
    raw_data_path=os.path.join('artifacts','raw.csv')
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    


### Data Ingestion Class: 
class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def intiate_data_ingestion(self):
        logging.info('Data Ingestion method is Start')
        
        try:
            df=pd.read_csv(os.path.join('notebooks/data','House_prices.csv'))
            logging.info('Dataset read as a pandas DataFrame')
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info('Train Test Split')
            
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)            
            
            logging.info('Ingestion of Data is Complted')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Some Error occured in Data ingestion Class")
            raise CustomException(e,sys)
        
        
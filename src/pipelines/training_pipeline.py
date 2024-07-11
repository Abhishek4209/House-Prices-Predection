import os 
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.intiate_data_ingestion()
    print(train_data_path,test_data_path)
    logging.info('Training is Complted')
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transfomration(train_data_path,test_data_path)
    
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr) 
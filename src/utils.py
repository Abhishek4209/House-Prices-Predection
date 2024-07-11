import os 
import sys
import numpy as np
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
        logging.info('Some Error occured into save object file path')
        
        
        
def evaluate_model(X_train, y_train,X_test, y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_test_pred=model.predict(X_test)
            
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score
                
        return report
    
    except Exception as e:
        logging.info('Some Error occured in Model evaluation process')
        raise CustomException(e,sys)
    
    
def load_objects(file_path):
    try:
        logging.info('Loading objects from file')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    
    
    except Exception as e:
        raise CustomException(e,sys)
        logging.info('Some Error occured in load_objects')
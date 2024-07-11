import numpy as np
import pandas as pd
import os
import sys
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression ,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor




@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Models Define')
            models={
            "Linear Regression": LinearRegression(),
            'Ridge Regression':Ridge(),
            'Lasso Regression':Lasso(),
            'Elasticnet Regression':ElasticNet(),
            "Decision Tree": DecisionTreeRegressor(),
            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5,p=2,algorithm="auto"),
            "Random Forest": RandomForestRegressor(),            
            'XGBRegressor':XGBRegressor(base_score=None, booster=None, callbacks=None,
            colsample_bylevel=None, colsample_bynode=None,
            colsample_bytree=None, device=None, early_stopping_rounds=None,
            enable_categorical=False, eval_metric=None, feature_types=None,
            gamma=None, grow_policy=None, importance_type=None,
            interaction_constraints=None, learning_rate=None, max_bin=None,
            max_cat_threshold=None, max_cat_to_onehot=None,
            max_delta_step=None, max_depth=None, max_leaves=None,
            min_child_weight=None, monotone_constraints=None,
            multi_strategy=None, n_estimators=32, n_jobs=None,
            num_parallel_tree=None, random_state=None)
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
    
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
import numpy as np
import pandas as pd
import os 
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object

## Data Transformation Config:
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    

## Data Transformation class :
class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation Intiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            numeric_features=['NoOfBedrooms', 'NoOfBathrooms', 'NoOfFloors',
            'FlatArea', 'LotArea', 'BasementArea', 'AreaOfTheHouseFromBasement',
            'LivingAreaAfterRenovation', 'LotAreaAfterRenovation', 'AgeOfHouse',
            'OverallGrade']
            
            categorical_features=['ConditionOfTheHouse']
            
            # Define the custom ranking for each ordinal variable
            condition_of_house=['Fair','Good','Excellent','Okay','Bad']
            
            logging.info('Pipeline Intiated')
            
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]
            )
            
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[condition_of_house])),
                ('scaler',StandardScaler())
                ]
            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numeric_features),
            ('cat_pipeline',cat_pipeline,categorical_features)
            
            ])
            
            return preprocessor
            logging.info('Pipeline Completed')
            
            
        except Exception as e:
            logging.info('Some Error occured into get_data_transformation_object')
            raise CustomException(e,sys)
        
            
    def initiate_data_transfomration(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            
            logging.info('Read Train and Test data Completed')
            logging.info(f'Train DataFrame head :  \n{train_df.head().to_string}')
            logging.info(f'Test DataFrame head :  \n{test_df.head().to_string}')
                
            logging.info('Obtaining preprocessor Object ')           

            preprocessing_obj=self.get_data_transformation_object()
                
            target_column_name="SalePrice"
            drop_columns=[target_column_name]
                
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]                
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]                
            
            ## apply the transformation 
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info('Applying preprocessing object on training and testing datsets.')
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(train_arr)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            
            
            logging.info('Preprocessing pickle in create and saved')
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
    
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Some Error Occured into initiate_data_transfomration')
            

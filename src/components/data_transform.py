import sys
import pandas as pd 
import numpy as np 
import os
from dataclasses import dataclass
from src.utils import save_obj


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.Exception import custom_exception
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig=DataTransformationConfig()
        
    def get_data_transformer(self):
        
        """
        data preprocessing phase
        """
        
        try:
            num_cols=['reading score', 'writing score']
            cat_cols=['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                    ]                
            )
            logging.info(f"numerical cols are {num_cols}")
            logging.info("numerical columns preprocessing completed")
            logging.info(f"categorical cols are {cat_cols}")
            
            logging.info("categorical columns preprocessing completed")
            
            preprocessor = ColumnTransformer(
                [ 
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            raise custom_exception(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Task 1: Reading data completed")
            logging.info("Task 2: Obtaining preprocessing object")
            
            preprocessing_obj= self.get_data_transformer()
            target_col_name="math score"
            
            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]
            
            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]
            
            
            
            
            logging.info("Training and test input features and targeted features created")
            
            # preprocessing input features using preprocessing object
            input_features_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_features_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("""
                         1) train _arry is created 
                         2)test arr is created 
                         Preprocessing completed ......100%
                         """)
            logging.info("file saving started")
            save_obj(
                
            file_path = self.DataTransformationConfig.preprocessor_obj_file_path,
            obj = preprocessing_obj
            )
            logging.info("file saving done")
            
            return (
                train_arr,
                test_arr,
                self.DataTransformationConfig.preprocessor_obj_file_path
            )
            

            
        except Exception as e:
            raise custom_exception(e,sys)
        
        
    

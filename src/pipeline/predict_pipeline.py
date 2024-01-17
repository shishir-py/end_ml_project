import sys
import os

import pandas as pd
import numpy as np 

from src.Exception import custom_exception

from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path=os.path.join("artifacts\model_pkl")
            preprocessor_path=os.path.join("artifacts\preprocessor.pkl")
            print(features)
            
            print("model Loading")
            model=load_object(file_path=model_path)
            print("model Loaded")
            print("preprocessor Loading")
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            scaled_data=preprocessor.transform(features)
            preds=model.predict(scaled_data)
            return preds
        except Exception as e:
            raise custom_exception(e, sys)
    
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def  get_data_as_df(self):
        try:
            test_input ={
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch': [self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }
            df= pd.DataFrame(test_input)
            print(df)
            return df

        except Exception as e:
            raise custom_exception(e, sys)
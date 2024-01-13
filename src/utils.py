import os
import sys
import pandas as pd
import numpy as np
# from src.Exception import custom_exception
from pickle4 import pickle
from src.Exception import custom_exception
from sklearn.metrics import r2_score


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            # print("done")
    except Exception as e:
        # print(f"Error: {e}")
        return custom_exception(e, sys)

# file_path = "artifacts/"
# obj = 1
# save_obj(file_path, obj)


def model_evaluation(X_train,y_train,X_test,y_test, models):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)
            y_train_preds = model.predict(X_train)
            y_test_preds = model.predict(X_test)
        
            
            train_model_score = r2_score(y_train, y_train_preds)
            test_model_score = r2_score(y_test, y_test_preds)
            
            report[list(models.keys())[i]]= test_model_score
            # print(report)
            
        return report
        # print(report)

    except Exception as e:
        raise custom_exception(e, sys)
   
        

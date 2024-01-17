import os
import sys
import pandas as pd
import numpy as np
from pickle4 import pickle
from src.Exception import custom_exception
from sklearn.metrics import r2_score
from src.logger import logging
import json
import dill
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



from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def model_evaluation(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        model_json={}
        for model_name, model in models.items():
            grid_search = GridSearchCV(model, params.get(model_name, {}), cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            # Get the best model from the hyperparameter tuning
            best_model = grid_search.best_estimator_
            # Make predictions on train and test sets
            y_train_preds = best_model.predict(X_train)
            y_test_preds = best_model.predict(X_test)
            
            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_preds)
            test_model_score = r2_score(y_test, y_test_preds)
            
            # Store the results in the report dictionary
            report[model_name] = {
                'model_info': best_model,
                'hyperparameters': params.get(model_name, {}),
                'train_score': train_model_score,
                'test_score': test_model_score
            }
            # Store the results in the model_json dictionary
            
            model_json[model_name] = {
            'model_name': model_name,
            'hyperparameters':params.get(model_name, {}),
            'train_score': report[model_name]['train_score'],
            'test_score': report[model_name]['test_score'],
            }
            # logging.info(report)
            logging.info(f"{model_name } - Model info: {best_model}, Train R2 Score: {train_model_score}, Test R2 Score: {test_model_score}")
            
        logging.info("json file dump started ...........")
        model_trained_path = os.path.join("artifacts", "model_report.json")
        
        with open (model_trained_path, 'w') as file:
            json.dump(model_json, file, indent = 4)
        logging.info("file dumped successfully")
        return report

    except Exception as e:
        raise custom_exception(e, sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise custom_exception(e, sys)
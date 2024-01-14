import os
import sys

from src.Exception import custom_exception
from src.logger import logging
#import dataclass
from dataclasses import dataclass
#import ml lib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
# import evaluation methods:


from src.utils import save_obj, model_evaluation

#config class
@dataclass
class ModelTrainerConfig:
    model_trained_path = os.path.join("artifacts", "model_pkl")
    
# model class
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            #splitting into train and test
            logging.info("splitting started")
            X_train,y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(verbose=False),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor()
            }
            model_params = {
            "LinearRegression": {},
            
            "DecisionTreeRegressor": {
            'max_depth': [5, 10, 20],  # Use a list of values for grid search
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
            },

            "RandomForestRegressor": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            "AdaBoostRegressor": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
            },
            
            "GradientBoostingRegressor": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            "CatBoostRegressor": {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'depth': [4, 6, 8, 10]
            },
            
            "KNeighborsRegressor": {
                'n_neighbors': [3, 5, 10],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
            
            "XGBRegressor": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'max_depth': [3, 5, 10],
                'min_child_weight': [1, 3, 5]
            }
}

            logging.info("Model Evaluation started")    
            
            model_report: dict = model_evaluation(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models,params=model_params)
            best_model_name = None
            best_test_score = float('-inf')
            for model_name, model_data in model_report.items():
                test_score = model_data['test_score']
                if test_score > best_test_score:
                    best_test_score = test_score
                    best_model_name = model_name

            best_model_info = model_report[best_model_name]['model_info']
            
            save_obj(
                file_path=self.model_trainer_config.model_trained_path,
                obj=best_model_info
            )
            logging.info(f"The best model has been saved")

            print(f"\nDetails of the model with the highest R2 test score:")
            print(f"Model Name: {model_report[best_model_name]['model_info']}")
 
            logging.info("model file saved")
        except Exception as e:
            raise custom_exception(e, sys)
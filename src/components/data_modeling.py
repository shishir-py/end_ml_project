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
                "Linear regression": LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "ADA Boost" : AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "catboost" : CatBoostRegressor(verbose=False),
                "k neighnnours": KNeighborsRegressor(),
                "XGboost": XGBRegressor()
            }
            logging.info("Model Evaluation started")    
            
            model_report: dict = model_evaluation(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models)
            best_score = max(sorted(model_report.values()))
            best_model = list(model_report.keys())[list(model_report.values()).index(best_score)]
            best_model=models[best_model]
            
            save_obj(
                file_path= self.model_trainer_config.model_trained_path,
                obj=best_model
            )
            logging.info("Model Evaluation done")
            logging.info(f"Best model is {best_model} and the r2_score is {best_score}")    
            print(f"Best model is {best_model} and the r2_score is {best_score}") 
        except Exception as e:
            raise custom_exception(e, sys)
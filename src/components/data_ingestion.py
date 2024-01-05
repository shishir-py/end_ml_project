import os
import sys
from src.Exception import custom_exception
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "raw_data.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("data Ingestion started")
        try:
            df=pd.read_csv("notebook\data\StudentsPerformance.csv")
            # we can chage data sorce by interchaing above script withe rit is from database or other sources
            logging.info('read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split initiated")
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv((self.ingestion_config.train_data_path),  index=False, header=True)
            test_set.to_csv((self.ingestion_config.test_data_path),  index=False, header=True)
            
            logging.info("Ingestion of the data is complited")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise custom_exception(e,sys)
        
if __name__== "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
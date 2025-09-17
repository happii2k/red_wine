import pandas as pd
from RED_WINE.entity.config_entity import ModelTrainerConfig
from RED_WINE.logging.logger import logger
from sklearn.linear_model import ElasticNet
import os
import joblib


class ModelTrainr:
    def __init__(self , config = ModelTrainerConfig):
        self.config = config

    def start_model_training(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_X = train_data.drop(self.config.target_column, axis=1)
        logger.info(f"Train x : {train_X.head}")
        train_y = train_data[self.config.target_column]

        test_y = test_data[self.config.target_column]

        lr  = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_X,train_y)
        joblib.dump(lr  , os.path.join(self.config.root_dir,self.config.model_name))


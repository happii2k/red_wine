import os
import sys
from RED_WINE.logging.logger import logger
from RED_WINE.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import pandas as pd

class DataTransformation:
    def __init__(self , config :DataTransformationConfig):
        self.config = config

    def start_data_transformation(self ):

        data = pd.read_csv(self.config.data_path)

        train , test  = train_test_split(data )

        train.to_csv(os.path.join(self.config.root_dir,"Train.csv"),index=False)
        logger.info("Train set is created")
        test.to_csv(os.path.join(self.config.root_dir,"Test.csv"),index=False)
        logger.info("Test set is created")
        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
        print(train.shape)
        print(test.shape)



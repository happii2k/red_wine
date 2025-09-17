import pandas as pd
from RED_WINE.config.configuration import ConfigurationManager
from RED_WINE.components.model_trainer import ModelTrainr 
from RED_WINE.logging.logger import logger

class ModelTrainerPipeline:
    def __init__(self):
        
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_trainer=ModelTrainr(config=model_trainer_config)
        model_trainer.start_model_training()

if __name__=="__main__":
    obj = ModelTrainerPipeline()
    obj.main()


from RED_WINE.config.configuration import ConfigurationManager
from RED_WINE.components.model_evaluate import ModelEvaluate
from RED_WINE.logging.logger import logger

class ModelEvaluatePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        logger.info("starting Pipeline")
        model_evaluate_conf = config.get_model_evaluation_config()
        logger.info("config has created")
        model_evaluate=ModelEvaluate(config=model_evaluate_conf)
        logger.info("Model evaluation starting")
        model_evaluate.start_model_evaluation()
        logger.info("Model evaluation ended")


if __name__ == "__main__":
    obj = ModelEvaluatePipeline()
    obj.main()
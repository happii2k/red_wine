from RED_WINE.components.data_transformation import DataTransformation
from RED_WINE.config.configuration import ConfigurationManager
from RED_WINE.logging.logger import logger
from pathlib import Path

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path(r"artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split()[-1]

                if status == "True":
                    config = ConfigurationManager()
                    logger.info("Configuration has started")
                    data_transf_confg=config.get_data_transformation_config()
                    logger.info("Configuration has ended")

                    DataTransformat=DataTransformation(config=data_transf_confg)
                    logger.info("Data Transformation has started")
                    DataTransformat.start_data_transformation()
                    logger.info("Data Transformation has ended")
                else:
                    raise Exception ("Your data schema is not valid")
                
        except Exception as e:
            print(e)
if __name__ == "__main__":
    obj = DataTransformationPipeline()
    obj.main()
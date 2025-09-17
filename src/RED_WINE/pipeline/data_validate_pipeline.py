from RED_WINE.config.configuration import ConfigurationManager

from RED_WINE.components.data_validation import DataValidation
from RED_WINE.logging.logger import logger

STAGE_NAME = "Data validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        data_validate_config  =config.get_data_validation_config()
        validate_data = DataValidation(config=data_validate_config)
        validate_data.validate_data()


if __name__ =="__main__":
    obj = DataValidationPipeline()
    obj.main()
        


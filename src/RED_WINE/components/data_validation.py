import pandas as pd
from RED_WINE.utils.utils import read_yaml
from RED_WINE.logging.logger import logger
from RED_WINE.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config:DataValidationConfig ):
        self.config = config
        
    def validate_data(self) -> bool :
        try:
            logger.info("Starting Validation")
            validation_status = None
            data = pd.read_csv(self.config.unzip_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_schema:
                if col not in all_cols:
                    validation_status = False
                    with open(self.config.STATUS_FILE,"w") as f:
                        f.write(f"Validation status: {validation_status}")
                else :
                    validation_status = True
                    with open(self.config.STATUS_FILE,"w") as f:
                        f.write(f"Validation status: {validation_status}")
            logger.info("Validation Completed")

            return validation_status
        
        except Exception as e:
            raise e







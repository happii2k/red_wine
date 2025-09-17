from pathlib import Path
from RED_WINE.utils.utils import read_yaml, create_directories
from RED_WINE.entity.config_entity import DataIngestionConfig ,DataValidationConfig , DataTransformationConfig , ModelTrainerConfig ,ModelEvaluateConfig
from RED_WINE.constants.constant import *
from RED_WINE.logging.logger  import logger


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH
    ):
        # Load YAML configs
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Ensure artifacts root exists
        create_directories([Path(self.config.artifacts_root)])
        logger.info("ConfigurationManager initialized successfully.")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Prepare and return DataIngestionConfig object."""
        config = self.config.data_ingestion

        # Ensure ingestion root dir exists
        create_directories([Path(config.root_dir)])
        logger.info(f"Data ingestion directories created at: {config.root_dir}")

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        logger.info("DataIngestionConfig created successfully.")
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:


        config = self.config.data_validation 
        schema = self.schema.COLUMNS

        create_directories([Path(config.root_dir)])
        logger.info(f"Data validation directories created at: {config.root_dir}")

        Data_Validation_Config  = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            unzip_dir=Path(config.unzip_dir),
            all_schema=schema
        )
        return Data_Validation_Config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config  = self.config.data_transformation

        create_directories([Path(config.root_dir)])

        data_transformation_config =  DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path)
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config  = self.config.model_trainer
        schema = self.schema.TARGET_COLUMN
        params = self.params.ElasticNet

        create_directories([Path(config.root_dir)])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            alpha = params.alpha,
            l1_ratio = params.l1_ratio,
            target_column = schema.name
            
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluateConfig:

        config = self.config.model_evaluate
        schema = self.schema.TARGET_COLUMN

        create_directories([Path(config.root_dir)]) 

        model_eval_confg = ModelEvaluateConfig(
            root_dir= config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metric_file_name= config.metric_file_name,
            target_column_name= schema.name

        )
        return model_eval_confg



from RED_WINE.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from RED_WINE.pipeline.data_transformation_pipeline import DataTransformationPipeline
from RED_WINE.pipeline.data_validate_pipeline import DataValidationPipeline
from RED_WINE.pipeline.model_evaluate_pipeline import ModelEvaluatePipeline
from RED_WINE.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from RED_WINE.logging.logger import logger

STAGE_NAME= "DATA INGESTION"
try:
    logger.info(f">>>> stage name : {STAGE_NAME} started <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME= "DATA VALIDATION"
try:
    logger.info(f">>>> stage name : {STAGE_NAME} started <<<<")
    obj = DataValidationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME= "DATA TRANSFORMATION"
try:
    logger.info(f">>>> stage name : {STAGE_NAME} started <<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME= "MODEL TRAINER"
try:
    logger.info(f">>>> stage name : {STAGE_NAME} started <<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME= "MODEL EVALUATER"
try:
    logger.info(f">>>> stage name : {STAGE_NAME} started <<<<")
    obj = ModelEvaluatePipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e




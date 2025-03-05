from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig, 
    DataTransformationConfig,
    ModelTrainerConfig
)

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

import sys

if __name__ == "__main__":
    try:
        # Getting the training configuration
        training_pipeline_config = TrainingPipelineConfig()
        # Create a DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        # Create a DataIngestion object
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiate the data ingestion process")

        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion process completed")
        print(data_ingestion_artifact)

        # Create a DataValidationConfig object
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)

        logging.info("Initiate the data validation process")
        data_validation_artifact = data_validation.initiate_data_validation()        
        logging.info("Data Validation process completed")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)

        logging.info("Initiate the data transformation process")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation process completed")
        print(data_transformation_artifact)

        logging.info("Model training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training artifact created")


    except Exception as e:
        raise NetworkSecurityException(e, sys)
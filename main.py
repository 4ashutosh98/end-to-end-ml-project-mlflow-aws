from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
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

        data_ingestion_artifact = data_ingestion.initate_data_ingestion()
        print(data_ingestion_artifact)


    except Exception as e:
        raise NetworkSecurityException(e, sys)
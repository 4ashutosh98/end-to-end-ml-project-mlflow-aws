from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, SAVED_FILE_NAME
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            return self.model.predict(x_transform)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
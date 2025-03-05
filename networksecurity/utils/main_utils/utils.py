import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys
import os
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path: str) -> dict:
    """
    This function reads the yaml file and returns the dictionary
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False):
    """
    This function writes the dictionary to the yaml file
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    This function saves the numpy array data to the file
    file_path : str : location to save the numpy array file
    array : np.array : numpy array to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    This function saves the object to the file
    file_path : str : location to save the object file
    obj : object : object to save
    """
    try:
        logging.info(f"Entered save_object method of main utils module")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        logging.info(f"Exited save_object method of main utils module")

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path : str) -> object:
    """
    This function loads the object from the file
    file_path : str : location to load the object file
    return : object : loaded object
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File {file_path} not found")
        with open(file_path, "rb") as file:
            print(file)
            return pickle.load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    This function loads the numpy array data from the file
    file_path : str : location to load the numpy array file
    return : np.array : loaded numpy array
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"Numpy array file {file_path} not found")
        with open(file_path, "rb") as file:
            return np.load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params) -> dict:
    """
    This function evaluates the models on the given data and returns the metrics
    X_train : np.array : training data
    y_train : np.array : training target
    X_test : np.array : testing data
    y_test : np.array : testing target
    models : dict : dictionary of models
    params : dict : dictionary of parameters
    return : dict : dictionary of metrics
    """
    try:
        report = {}
        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]
            model_params = params[list(models.keys())[i]]

            gs = GridSearchCV(model, model_params, cv=3, verbose=1, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) 

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
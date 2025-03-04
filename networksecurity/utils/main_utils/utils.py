import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys
import os
import numpy as np
import pickle

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
    
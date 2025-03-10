import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occured in python script name [{self.filename}] line number [{self.lineno}] error message [{self.error_message}]"
    
if __name__ == "__main__":
    try:
        logger.logging.info("Enter the try block")
        a = 1/0
        print("This will not be printed", a)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
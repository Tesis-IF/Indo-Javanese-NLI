from decouple import config
from datetime import datetime
import logging
import os

def init_logger():
    FORMAT = '%(levelname)s: %(asctime)s %(message)s'
    formatter = logging.Formatter(FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    # Add a handler to the logger
    if not os.path.exists("logger/logs/"):
        os.makedirs("logger/logs/")
    log_filename = "logger/logs/log_" + str(datetime.now().strftime("%Y%m%d")) + ".txt"

    logging.FileHandler(log_filename)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger(config("LOGGER_NAME"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("init_logger: Starting logger...")

    return logger
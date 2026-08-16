import logging
import sys


def get_logger(name):
    logger=logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    console_handler=logging.StreamHandler(sys.stdout)
    log_format=logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",)
        
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.propagate=False
    return logger
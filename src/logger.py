import datetime
import logging
import multiprocessing
from logging.handlers import QueueListener

def setup_logger():
    timestamp = str(datetime.datetime.now())[:19].replace(":", "-")

    log_queue = multiprocessing.Queue()
    logger = logging.getLogger("main_logger")

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'./logs/run-{timestamp}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    listener = QueueListener(log_queue, file_handler)
    listener.start()
    return logger, log_queue

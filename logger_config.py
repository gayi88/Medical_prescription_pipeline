import logging

def setup_logger(name, log_file='app.log', level=logging.DEBUG):
    """
    Creates and returns a logger object
    Logs messages to a file
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

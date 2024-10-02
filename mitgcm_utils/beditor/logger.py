# logger_setup.py
import logging
import sys

# from logging.handlers import RotatingFileHandler


def setup_logging(level=logging.INFO):  # type: ignore
    """
    Configures the root logger to log messages to both stdout and a rotating log file.

    Parameters:
    - level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logger = logging.getLogger()  # Root logger
    logger.setLevel(level)

    # Check if handlers are already added to avoid duplication
    if not logger.handlers:
        # StreamHandler for stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)

        # RotatingFileHandler for log file
        # fh = RotatingFileHandler("app.log", maxBytes=5 * 1024 * 1024, backupCount=5)
        # fh.setLevel(level)

        # Define a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        sh.setFormatter(formatter)
        # fh.setFormatter(formatter)

        # Add handlers to the root logger
        logger.addHandler(sh)
        # logger.addHandler(fh)

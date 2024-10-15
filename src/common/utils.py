import os
import yaml
import time
import logging
from logging.config import dictConfig


def get_logger(name: str) -> logging.Logger:
    """
    Loads the logging configuration from a YAML file and returns an instance of
    a logger.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        An instance of the logger.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("This is an info message.")
    """

    # Get the current directory
    current_dir = os.path.dirname(__file__)

    # The root directory is 2 levels up
    root_dir = os.path.abspath(os.path.join(current_dir, "..", "..")) + "/"

    with open(root_dir + "logging.yaml", "r") as f:
        config = yaml.safe_load(f.read())
        dictConfig(config)

    return logging.getLogger(name)


class Timer:
    """
    A simple timer class similar to MATLAB's tic and toc functions.

    Methods
    -------
    tic():
        Starts the timer.

    toc() -> str:
        Stops the timer and returns the elapsed time in HH:MM:SS format.
    """

    def __init__(self):
        self.start_time = None
        self.tic()

    def tic(self):
        """Starts the timer."""
        self.start_time = time.perf_counter()

    def toc(self) -> str:
        """
        Stops the timer and returns the elapsed time in HH:MM:SS format.

        Returns
        -------
        str
            The elapsed time in HH:MM:SS format.
        """
        if self.start_time is None:
            raise ValueError(
                "Timer has not been started. Call tic() to start the timer."
            )

        elapsed_time = time.perf_counter() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

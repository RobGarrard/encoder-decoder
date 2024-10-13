import os
import yaml
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

"""Meta-learning implementation with MAML."""

__version__ = "0.1.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from . import data
from . import models
from . import metrics
from . import train
from . import utils
from . import viz

__all__ = [
    "data",
    "models", 
    "metrics",
    "train",
    "utils",
    "viz"
]

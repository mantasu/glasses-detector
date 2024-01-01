from enum import Enum, auto
from typing import Collection, TypeVar
from urllib.parse import urlparse

import numpy as np
from PIL import Image

T = TypeVar("T")
type Scalar = bool | int | float | str
type FormattedPred = Scalar | Collection[Scalar] | np.ndarray | Image.Image


class PredType(Enum):
    SCALAR = auto()
    ARRAY = auto()
    IMAGE = auto()
    OTHER = auto()


def pred_type(pred: FormattedPred | T) -> PredType:
    # Define scalar types
    scalar_types = (bool, int, float, str)

    # Get the type of the prediction
    if isinstance(pred, scalar_types):
        return PredType.SCALAR
    elif isinstance(pred, Collection) and all(
        isinstance(item, scalar_types) for item in pred
    ):
        return PredType.ARRAY
    elif isinstance(pred, Image.Image):
        return PredType.IMAGE
    else:
        return PredType.OTHER


def is_url(x):
    # https://stackoverflow.com/a/38020041
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False

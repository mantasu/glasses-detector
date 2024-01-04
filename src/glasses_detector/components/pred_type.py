from enum import Enum
from typing import Any, Iterable, Self, TypeGuard, TypeVar

import numpy as np
import torch
from PIL import Image

type Scalar = bool | int | float | str | np.generic | np.ndarray | torch.Tensor
type Tensor = Iterable[Scalar | Tensor] | Image.Image
type Default = Scalar | Tensor
type StandardScalar = bool | int | float | str
type StandardTensor = list[StandardScalar | StandardTensor]
type StandardDefault = StandardScalar | StandardTensor
type NonDefault = TypeVar("NonDefault")
type Custom = Default | NonDefault


class PredType(Enum):
    """Enum class for expected prediction types.

    This class specifies the expected prediction types for mainly
    classification, detection and segmentation models that work with
    image data. The expected types are called **Default**.

    Type Categories
    ---------------

    There are two categories of **Default** prediction types:

    1. **Standard**: these are the basic *Python* types, i.e.,
       :class:`bool`, :class:`int`, :class:`float`, :class:`str`, and
       :class:`list`. Standard types are easy to work with, e.g., they
       can be parsed by JSON and YAML formats.
    2. **Non-standard**: these additionally contain types like
       :class:`numpy.ndarray`, :class:`torch.Tensor`, and
       :class:`PIL.Image.Image`. They are convenient because due to more
       flexibility for model prediction outputs. In most of the cases,
       they can be converted to standard types via :meth:`standardize`.

    Examples
    --------

    :class:`PredType` constants can be used to specify the expected
    types when defining the methods:

        >>> def predict_class(
        ...     self,
        ...     image: Image.Image,
        ...     output_format: str = "score",
        ... ) -> PredType.StandardScalar:
        ...     pass

    :class:`PredType` static and class methods can be used to check
    the type of the prediction:

        >>> PredType.is_standard_scalar(1)
        True
        >>> PredType.is_standard_scalar(np.array([1])[0])
        False
        >>> PredType.is_default(Image.fromarray(np.zeros((1, 1))))
        True

    Finally, :meth:`standardize` can be used to convert the
    prediction to a standard type:

        >>> PredType.standardize(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> PredType.standardize(Image.fromarray(np.zeros((1, 1))))
        [[0.0]]
    """

    @staticmethod
    def is_scalar(pred: Any) -> TypeGuard[Scalar]:
        return isinstance(pred, (bool, int, float, str, np.generic)) or (
            isinstance(pred, (torch.Tensor, np.ndarray)) and pred.ndim == 0
        )

    @staticmethod
    def is_standard_scalar(pred: Any) -> TypeGuard[StandardScalar]:
        return isinstance(pred, (bool, int, float, str))

    @classmethod
    def is_tensor(cls, pred: Any) -> TypeGuard[Tensor]:
        return isinstance(pred, Image.Image) or (
            isinstance(pred, Iterable)
            and not cls.is_scalar(pred)
            and all([cls.is_scalar(p) or cls.is_tensor(p) for p in pred])
        )

    @classmethod
    def is_standard_tensor(cls, pred: Any) -> TypeGuard[StandardTensor]:
        return isinstance(pred, list) and all(
            [cls.is_standard_scalar(p) or cls.is_standard_tensor(p) for p in pred]
        )

    @classmethod
    def is_default(cls, pred: Any) -> TypeGuard[Default]:
        return cls.is_scalar(pred) or cls.is_tensor(pred)

    @classmethod
    def is_standard_default(cls, pred: Any) -> TypeGuard[StandardDefault]:
        return cls.is_standard_scalar(pred) or cls.is_standard_tensor(pred)

    @classmethod
    def check(cls, pred: Any) -> Self:
        if cls.is_standard_scalar(pred):
            return cls.StandardScalar
        elif cls.is_scalar(pred):
            return cls.Scalar
        elif cls.is_standard_tensor(pred):
            return cls.StandardTensor
        elif cls.is_tensor(pred):
            return cls.Tensor
        else:
            return cls.NonDefault

    @classmethod
    def standardize(cls, pred: Default) -> StandardDefault:
        if isinstance(pred, (bool, int, float, str)):
            return pred
        elif isinstance(pred, np.generic):
            return cls.standardize(pred.item())
        elif isinstance(pred, (np.ndarray, torch.Tensor)) and pred.ndim == 0:
            return cls.standardize(pred.item())
        elif isinstance(pred, Image.Image):
            return np.asarray(pred).tolist()
        elif isinstance(pred, Iterable):
            return [cls.standardize(item) for item in pred]
        else:
            raise ValueError(f"Cannot standardize {type(pred)}")

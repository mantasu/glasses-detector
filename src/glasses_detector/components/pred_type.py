from enum import Enum, auto
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
type NonDefault = TypeVar("NonDefault", bound=Any)
type Anything = Default | NonDefault


class PredType(Enum):
    """Enum class for expected prediction types.

    This class specifies the expected prediction types for mainly
    classification, detection and segmentation models that work with
    image data. The expected types are called **Default**.

    Note:
        The constants defined in this class are only only enums, not
        actual classes or :class:`type` objects. For type hints,
        corresponding type aliases are defined in the same file as this
        class.

    Warning:
        The enum types are not exclusive, for example, :attr:`SCALAR`
        is also a :attr:`DEFAULT`.

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

    Types
    -----

    The following prediction types/categories are defined in this class:

        1. :attr:`SCALAR`: see :meth:`is_scalar`.
        2. :attr:`TENSOR`: see :meth:`is_tensor`.
        3. :attr:`DEFAULT`: see :meth:`is_default`.
        4. :attr:`STANDARD_SCALAR`: see :meth:`is_standard_scalar`.
        5. :attr:`STANDARD_TENSOR`: see :meth:`is_standard_tensor`.
        6. :attr:`STANDARD_DEFAULT`: see :meth:`is_standard_default`.
        7. :attr:`NON_DEFAULT`: see :meth:`check`.

    Examples
    --------

    Type aliases (:class:`type` objects) corresponding to the enums of
    this class are defined in the same file as :class:`PredType`. They
    can be used to specify the expected types when defining the methods:

        >>> def predict_class(
        ...     self,
        ...     image: Image.Image,
        ...     output_format: str = "score",
        ... ) -> StandardScalar:
        ...     ...

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

    SCALAR = auto()
    TENSOR = auto()
    DEFAULT = auto()
    STANDARD_SCALAR = auto()
    STANDARD_TENSOR = auto()
    STANDARD_DEFAULT = auto()
    NON_DEFAULT = auto()

    @staticmethod
    def is_scalar(pred: Any) -> TypeGuard[Scalar]:
        """Check if the prediction is a scalar.

        Checks if the given value is a scalar, i.e., one of the
        following types:

            * :class:`bool`
            * :class:`int`
            * :class:`float`
            * :class:`str`
            * :class:`numpy.generic`
            * :class:`numpy.ndarray` with ``ndim == 0``
            * :class:`torch.Tensor` with ``ndim == 0``

        Args:
            pred (typing.Any): The value to check.

        Returns:
            typing.TypeGuard[Scalar]: ``True`` if the value is a scalar,
            ``False`` otherwise.
        """
        return isinstance(pred, (bool, int, float, str, np.generic)) or (
            isinstance(pred, (torch.Tensor, np.ndarray)) and pred.ndim == 0
        )

    @staticmethod
    def is_standard_scalar(pred: Any) -> TypeGuard[StandardScalar]:
        """Check if the prediction is a standard scalar.

        Checks if the given value is a standard scalar, i.e., one of the
        following types:

            * :class:`bool`
            * :class:`int`
            * :class:`float`
            * :class:`str`

        Args:
            pred (typing.Any): The value to check.

        Returns:
            typing.TypeGuard[StandardScalar]: ``True`` if the value is a
            standard scalar, ``False`` otherwise.
        """
        return isinstance(pred, (bool, int, float, str))

    @classmethod
    def is_tensor(cls, pred: Any) -> TypeGuard[Tensor]:
        """Check if the prediction is a tensor.

        Checks if the given value is a tensor, i.e., one of the
        following types:

            * :class:`PIL.Image.Image`
            * :class:`Iterable` of scalars or tensors, including lists,
              tuples, :class:`numpy.ndarray` and :class:`torch.Tensor`
              objects and other iterables.

        Args:
            pred (typing.Any): The value to check.

        Returns:
            typing.TypeGuard[Tensor]: ``True`` if the value is a tensor,
            ``False`` otherwise.
        """
        return isinstance(pred, Image.Image) or (
            isinstance(pred, Iterable)
            and not cls.is_scalar(pred)
            and all([cls.is_scalar(p) or cls.is_tensor(p) for p in pred])
        )

    @classmethod
    def is_standard_tensor(cls, pred: Any) -> TypeGuard[StandardTensor]:
        """Check if the prediction is a standard tensor.

        Checks if the given value is a standard tensor, i.e., one of the
        following types:

            * :class:`list` of standard scalars or standard tensors. No
              other iterables than lists are allowed.

        Args:
            pred (typing.Any): The value to check.

        Returns:
            typing.TypeGuard[StandardTensor]: ``True`` if the value is a
            standard tensor, ``False`` otherwise.
        """
        return isinstance(pred, list) and all(
            [cls.is_standard_scalar(p) or cls.is_standard_tensor(p) for p in pred]
        )

    @classmethod
    def is_default(cls, pred: Any) -> TypeGuard[Default]:
        """Check if the prediction is a default type.

        Checks if the given value is a default type, i.e., one of the
        following types:

            * All the types defined in :meth:`is_scalar`.
            * All the types defined in :meth:`is_tensor`.

        Args:
            pred (typing.Any): The value to check.

        Returns:
            typing.TypeGuard[Default]: ``True`` if the value is a
            default type, ``False`` otherwise.
        """
        return cls.is_scalar(pred) or cls.is_tensor(pred)

    @classmethod
    def is_standard_default(cls, pred: Any) -> TypeGuard[StandardDefault]:
        """Check if the prediction is a standard default type.

        Checks if the given value is a standard default type, i.e., one
        of the following types:

            * All the types defined in :meth:`is_standard_scalar`.
            * All the types defined in :meth:`is_standard_tensor`.

        Args:
            pred (typing.Any): The value to check.

        Returns:
            typing.TypeGuard[StandardDefault]: ``True`` if the value is
            a standard default type, ``False`` otherwise.
        """
        return cls.is_standard_scalar(pred) or cls.is_standard_tensor(pred)

    @classmethod
    def check(cls, pred: Any) -> Self:
        """Check the type of the prediction.

        Checks the type of the prediction and returns the corresponding
        enum of the lowest type category. First, it checks if the
        prediction is a standard scalar or a regular scalar (in that
        order). If not, it checks if the prediction is a standard tensor
        or a regular tensor (in that order). Finally, if none of the
        previous checks are successful, it returns :attr:`NON_DEFAULT`.

        Note:
            All four types, i.e., :attr:`STANDARD_SCALAR`,
            :attr:`SCALAR`, :attr:`STANDARD_TENSOR`, and :attr:`TENSOR`
            are subclasses of :attr:`DEFAULT`.

        Args:
            pred (typing.Any): The value to check.

        Returns:
            typing.Self: The corresponding enum of the lowest type
            category or :attr:`NON_DEFAULT` if no **default** category
            is applicable.
        """
        if cls.is_standard_scalar(pred):
            return cls.STANDARD_SCALAR
        elif cls.is_scalar(pred):
            return cls.SCALAR
        elif cls.is_standard_tensor(pred):
            return cls.STANDARD_TENSOR
        elif cls.is_tensor(pred):
            return cls.TENSOR
        else:
            return cls.NON_DEFAULT

    @classmethod
    def standardize(cls, pred: Default) -> StandardDefault:
        """Standardize the prediction.

        Standardize the prediction to a standard default type. If the
        prediction is already a standard default type, it is returned
        as-is. Otherwise, it is converted to a standard default type
        using the following rules:

            * :class:`bool`, :class:`int`, :class:`float`, and
              :class:`str` are returned as-is.
            * :class:`numpy.generic` and :class:`numpy.ndarray` with
              ``ndim == 0`` are converted to standard scalars.
            * :class:`torch.Tensor` with ``ndim == 0`` is converted to
              standard scalars.
            * :class:`numpy.ndarray` and :class:`torch.Tensor` with
              ``ndim > 0`` are converted to standard tensors.
            * :class:`PIL.Image.Image` is converted to standard tensors
              by converting it to a :class:`numpy.ndarray` and then
              applying the previous rule.
            * All other iterables are converted to standard tensors by
              applying the previous rule to each element.

        Args:
            pred (Default): The **default** prediction to standardize.

        Raises:
            ValueError: If the prediction cannot be standardized. This
            can if a prediction is not **default** or if a class, such
            as :class:`torch.Tensor` or :class:`numpy.ndarray` return
            a scalar that is not of type defined in
            :meth:`is_standard_scalar`.

        Returns:
            StandardDefault: The standardized prediction.
        """
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

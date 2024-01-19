"""
.. class:: Scalar

.. data:: Scalar
    :noindex:
    :type: typing.TypeAliasType
    :value: StandardScalar | np.generic | np.ndarray | torch.Tensor

    Type alias for a scalar prediction. For more information, see
    :attr:`~PredType.SCALAR`.

        :class:`bool` | :class:`int` | :class:`float` | :class:`str`
        | :class:`~numpy.generic` | :class:`~numpy.ndarray`
        | :class:`~torch.Tensor`

.. class:: Tensor

.. data:: Tensor
    :noindex:
    :type: typing.TypeAliasType
    :value: Iterable[Scalar | Tensor] | PIL.Image.Image

    Type alias for a tensor prediction. For more information, see
    :attr:`~PredType.TENSOR`.

        :class:`~typing.Iterable` | :class:`~PIL.Image.Image`

.. class:: Default

.. data:: Default
    :noindex:
    :type: typing.TypeAliasType
    :value: Scalar | Tensor

    Type alias for a default prediction. For more information, see
    :attr:`~PredType.DEFAULT`.

        :class:`bool` | :class:`int` | :class:`float` | :class:`str`
        | :class:`~numpy.generic` | :class:`~numpy.ndarray`
        | :class:`~torch.Tensor` | :class:`~typing.Iterable`
    
.. class:: StandardScalar

.. data:: StandardScalar
    :noindex:
    :type: typing.TypeAliasType
    :value: bool | int | float | str

    Type alias for a standard scalar prediction. For more information,
    see :attr:`~PredType.STANDARD_SCALAR`.

        :class:`bool` | :class:`int` | :class:`float` | :class:`str`

.. class:: StandardTensor

.. data:: StandardTensor
    :noindex:
    :type: typing.TypeAliasType
    :value: list[StandardScalar | StandardTensor]

    Type alias for a standard tensor prediction. For more information,
    see :attr:`~PredType.STANDARD_TENSOR`.

        :class:`list`

.. class:: StandardDefault

.. data:: StandardDefault
    :noindex:
    :type: typing.TypeAliasType
    :value: StandardScalar | StandardTensor

    Type alias for a standard default prediction. For more information,
    see :attr:`~PredType.STANDARD_DEFAULT`.

        :class:`bool` | :class:`int` | :class:`float` | :class:`str`
        | :class:`list`

.. class:: NonDefault[T]

.. data:: NonDefault[T]
    :noindex:
    :type: typing.TypeAliasType
    :value: T

    Type alias for a non-default prediction. For more information, see
    :attr:`~PredType.NON_DEFAULT`.

        :data:`~typing.Any`

.. class:: Either

.. data:: Either
    :noindex:
    :type: typing.TypeAliasType
    :value: Default | NonDefault

    Type alias for either default or non-default prediction, i.e., any
    prediction.

        :data:`~typing.Any`
"""
from enum import Enum, auto
from typing import Any, Iterable, Self, TypeGuard

import numpy as np
import torch
from PIL import Image

type Scalar = bool | int | float | str | np.generic | np.ndarray | torch.Tensor
type Tensor = Iterable[Scalar | Tensor] | Image.Image
type Default = Scalar | Tensor
type StandardScalar = bool | int | float | str
type StandardTensor = list[StandardScalar | StandardTensor]
type StandardDefault = StandardScalar | StandardTensor
type NonDefault[T] = T
type Anything = Default | NonDefault


class PredType(Enum):
    """Enum class for expected prediction types.

    This class specifies the expected prediction types mainly for
    classification, detection and segmentation models that work with
    image data. The expected types are called **Default** and there are
    two categories of them:

    1. **Standard**: these are the basic *Python* types, i.e.,
       :class:`bool`, :class:`int`, :class:`float`, :class:`str`, and
       :class:`list`. Standard types are easy to work with, e.g., they
       can be parsed by JSON and YAML formats.
    2. **Non-standard**: these additionally contain types like
       :class:`numpy.ndarray`, :class:`torch.Tensor`, and
       :class:`PIL.Image.Image`. They are convenient due to more
       flexibility for model prediction outputs. In most cases, they can
       be converted to standard types via :meth:`standardize`.

    Note:
        The constants defined in this class are only enums, not
        actual classes or :class:`type` objects. For type hints,
        corresponding type aliases are defined in the same file as this
        class.

    Warning:
        The enum types are not exclusive, for example, :attr:`SCALAR`
        is also a :attr:`DEFAULT`.

    Examples
    --------

    Type aliases (:class:`~typing.TypeAliasType` objects defined
    using :class:`type` keyword) corresponding to the enums of
    this class are defined in the same file as :class:`PredType`. They
    can be used to specify the expected types when defining the methods:

        >>> from glasses_detector.components.pred_type import StandardScalar
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

    """Enum: Scalar type. A prediction is considered to be a scalar if"""

    SCALAR = auto()
    """
    PredType: Scalar type. A prediction is considered to be a scalar if
    it is one of the following types:

    * :class:`bool`
    * :class:`int`
    * :class:`float`
    * :class:`str`
    * :class:`numpy.generic`
    * :class:`numpy.ndarray` with ``ndim == 0``
    * :class:`torch.Tensor` with ``ndim == 0``

    :meta hide-value:
    """

    TENSOR = auto()
    """
    PredType: Tensor type. A prediction is considered to be a tensor if
    it is one of the following types:
    
    * :class:`PIL.Image.Image`
    * :class:`~typing.Iterable` of scalars or tensors of any iterable
      type, including :class:`list`, :class:`tuple`,
      :class:`~typing.Collection`, :class:`numpy.ndarray` and
      :class:`torch.Tensor` objects, and any other iterables.

    :meta hide-value:
    """

    DEFAULT = auto()
    """
    PredType: Default type. A prediction is considered to be a default
    type if it is one of the following types:
    
    * Any of the types defined in :attr:`SCALAR`.
    * Any of the types defined in :attr:`TENSOR`.
    
    :meta hide-value:
    """

    STANDARD_SCALAR = auto()
    """
    PredType: Standard scalar type. A prediction is considered to be a
    standard scalar if it is one of the following types:
    
    * :class:`bool`
    * :class:`int`
    * :class:`float`
    * :class:`str`
    
    :meta hide-value:
    """

    STANDARD_TENSOR = auto()
    """
    PredType: Standard tensor type. A prediction is considered to be a
    standard tensor if it is one of the following types:
        
    * :class:`list` of standard scalars or standard tensors. No other
      iterables than lists are allowed.
    
    :meta hide-value:
    """

    STANDARD_DEFAULT = auto()
    """
    PredType: Standard default type. A prediction is considered to be a
    standard default type if it is one of the following types:
    
    * Any of the types defined in :attr:`STANDARD_SCALAR`.
    * Any of the types defined in :attr:`STANDARD_TENSOR`.
    
    :meta hide-value:
    """

    NON_DEFAULT = auto()
    """
    PredType: Non-default type. A prediction is considered to be a
    non-default type if it is not a default type, i.e., it is not any of
    the types defined in :attr:`DEFAULT`.

    :meta hide-value:
    """

    @staticmethod
    def is_scalar(pred: Any) -> TypeGuard[Scalar]:
        """Checks if the prediction is a **scalar**.

        .. seealso::

            :attr:`SCALAR`

        Args:
            pred: The value to check.

        Returns:
            :data:`True` if the value is a **scalar**, :data:`False`
            otherwise.
        """
        return isinstance(pred, (bool, int, float, str, np.generic)) or (
            isinstance(pred, (torch.Tensor, np.ndarray)) and pred.ndim == 0
        )

    @staticmethod
    def is_standard_scalar(pred: Any) -> TypeGuard[StandardScalar]:
        """Checks if the prediction is a **standard scalar**.

        .. seealso::

            :attr:`STANDARD_SCALAR`

        Args:
            pred: The value to check.

        Returns:
            :data:`True` if the value is a **standard scalar**,
            :data:`False` otherwise.
        """
        return isinstance(pred, (bool, int, float, str))

    @classmethod
    def is_tensor(cls, pred: Any) -> TypeGuard[Tensor]:
        """Checks if the prediction is a **tensor**.

        .. seealso::

            :attr:`TENSOR`

        Args:
            pred: The value to check.

        Returns:
            :data:`True` if the value is a **tensor**,
            :data:`False` otherwise.
        """
        return isinstance(pred, Image.Image) or (
            isinstance(pred, Iterable)
            and not cls.is_scalar(pred)
            and all([cls.is_scalar(p) or cls.is_tensor(p) for p in pred])
        )

    @classmethod
    def is_standard_tensor(cls, pred: Any) -> TypeGuard[StandardTensor]:
        """Checks if the prediction is a **standard tensor**.

        .. seealso::

            :attr:`STANDARD_TENSOR`

        Args:
            pred: The value to check.

        Returns:
            :data:`True` if the value is a **standard tensor**,
            :data:`False` otherwise.
        """
        return isinstance(pred, list) and all(
            [cls.is_standard_scalar(p) or cls.is_standard_tensor(p) for p in pred]
        )

    @classmethod
    def is_default(cls, pred: Any) -> TypeGuard[Default]:
        """Checks if the prediction is a **default** type.

        .. seealso::

            :attr:`DEFAULT`

        Args:
            pred: The value to check.

        Returns:
            :data:`True` if the type of the value is **default**,
            :data:`False` otherwise.
        """
        return cls.is_scalar(pred) or cls.is_tensor(pred)

    @classmethod
    def is_standard_default(cls, pred: Any) -> TypeGuard[StandardDefault]:
        """Checks if the prediction is a **standard default** type.

        .. seealso::

            :attr:`STANDARD_DEFAULT`

        Args:
            pred: The value to check.

        Returns:
            :data:`True` if the type of the value is **standard
            default**, :data:`False` otherwise.
        """
        return cls.is_standard_scalar(pred) or cls.is_standard_tensor(pred)

    @classmethod
    def check(cls, pred: Any) -> Self:
        """Checks the type of the prediction and returns its enum.

        Checks the type of the prediction and returns the corresponding
        enum of the lowest type category. First, it checks if the
        prediction is a **standard scalar** or a regular **scalar** (in
        that order). If not, it checks if the prediction is a **standard
        tensor** or a regular **tensor** (in that order). Finally, if
        none of the previous checks are successful, it returns
        :attr:`NON_DEFAULT`.

        Note:
            All four types, i.e., :attr:`STANDARD_SCALAR`,
            :attr:`SCALAR`, :attr:`STANDARD_TENSOR`, and :attr:`TENSOR`
            are subclasses of :attr:`DEFAULT`.

        Args:
            pred: The value to check.

        Returns:
            The corresponding enum of the lowest type category or
            :attr:`NON_DEFAULT` if no **default** category is
            applicable.
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

        Standardize the prediction to a **standard default** type. If
        the prediction is already a **standard default** type, it is
        returned as-is. Otherwise, it is converted to a **standard
        default** type using the following rules:

        * :class:`bool`, :class:`int`, :class:`float`, and :class:`str`
          are returned as-is.
        * :class:`numpy.generic` and :class:`numpy.ndarray` with
          :attr:`~numpy.ndarray.ndim` = ``0`` are converted to
          **standard scalars**.
        * :class:`torch.Tensor` with :attr:`~numpy.ndarray.ndim` = ``0``
          is converted to **standard scalars**.
        * :class:`numpy.ndarray` and :class:`torch.Tensor` with
          :attr:`~numpy.ndarray.ndim` > ``0`` are converted to
          **standard tensors**.
        * :class:`PIL.Image.Image` is converted to **standard tensors**
          by converting it to a :class:`numpy.ndarray` and then
          applying the previous rule.
        * All other iterables are converted to **standard tensors** by
          applying the previous rule to each element.

        Args:
            pred: The **default** prediction to standardize.

        Raises:
            ValueError: If the prediction cannot be standardized. This
                can happen if a prediction is not **default** or if a
                class, such as :class:`torch.Tensor` or
                :class:`numpy.ndarray` returns a scalar that is not of
                type defined in :attr:`SCALAR`.

        Returns:
            The standardized prediction.
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

"""
.. class:: FilePath

.. data:: FilePath
    :noindex:
    :type: typing.TypeAliasType
    :value: str | bytes | os.PathLike

    Type alias for a file path.

        :class:`str` | :class:`bytes` | :class:`os.PathLike`
"""

import functools
import imghdr
import os
import typing
from typing import Any, Callable, Concatenate, Iterable, TypeGuard, overload
from urllib.parse import urlparse

import torch

type FilePath = str | bytes | os.PathLike


class copy_signature[F]:
    """Decorator to copy a function's signature.

    This decorator takes a function and copies its signature to the
    decorated function. This is useful when you want to copy the
    signature of a function to a wrapper function.

    Example
    -------

    .. code-block:: python

        def full_signature(x: bool, *extra: int) -> str: ...

        @copy_signature(full_signature)
        def test_signature(*args, **kwargs):
            return full_signature(*args, **kwargs)

        reveal_type(test_signature)  # 'def (x: bool, *extra: int) -> str'

    .. seealso::

        https://stackoverflow.com/a/59717891

    Args:
        target: The function whose signature to copy.
    """

    def __init__(self, target: F) -> None: ...

    def __call__(self, wrapped: Callable[..., Any]) -> F: ...


def copy_method_signature[
    **P, T
](source: Callable[Concatenate[Any, P], T]) -> Callable[
    [Callable], Callable[Concatenate[Any, P], T]
]:
    """Decorator to copy a method's signature.

    This decorator takes a class method and copies its signature to the
    decorated method. This is useful when you want to copy the
    signature of a method to a wrapper method.

    Example
    -------

    .. code-block:: python

        def full_signature(x: bool, *extra: int) -> str: ...

        @copy_method_signature(full_signature)
        def test_signature(*args, **kwargs):
            return full_signature(*args, **kwargs)

        reveal_type(test_signature)  # 'def (x: bool, *extra: int) -> str'

    .. seealso::

        https://github.com/python/typing/issues/270#issuecomment-1344537820

    Args:
        source (typing.Callable[typing.Concatenate[typing.Any, P], T]):
            The method whose signature to copy.

    Returns:
        typing.Callable[[typing.Callable], typing.Callable[typing.Concatenate[typing.Any, P], T]]:
            The decorator to copy the signature.
    """

    def wrapper(target: Callable) -> Callable[Concatenate[Any, P], T]:
        @functools.wraps(source)
        def wrapped(self, *args: P.args, **kwargs: P.kwargs) -> T:
            return target(self, *args, **kwargs)

        return wrapped

    return wrapper


class eval_infer_mode:
    """Context manager and decorator for evaluation and inference.

    This class can be used as a context manager or a decorator to set a
    PyTorch :class:`~torch.nn.Module` to evaluation mode via
    :meth:`~torch.nn.Module.eval` and enable
    :class:`~torch.inference_mode` for the duration of a function
    or a ``with`` statement. After the function or the ``with``
    statement, the model's mode, i.e., :attr:`~torch.nn.Module.training`
    property, and :class:`~torch.inference_mode` are restored to their
    original states.

    Example
    -------

    .. code-block:: python

        model = ...  # Your PyTorch model

        @eval_infer_mode(model)
        def your_function():
            # E.g., forward pass
            pass

        # or

        with eval_infer_mode(model):
            # E.g., forward pass
            pass

    Args:
        model (torch.nn.Module): The PyTorch model to be set to
            evaluation mode.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.was_training = model.training

    def __call__[F](self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self.model.eval()
        self._inference_mode = torch.inference_mode(True)
        self._inference_mode.__enter__()

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self._inference_mode.__exit__(type, value, traceback)
        self.model.train(self.was_training)


def is_url(x: str) -> bool:
    """Check if a string is a valid URL.

    Takes any string and checks if it is a valid URL.

    .. seealso::

        https://stackoverflow.com/a/38020041

    Args:
        x: The string to check.

    Returns:
       :data:`True` if the string is a valid URL, :data:`False`
       otherwise.
    """
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


@overload
def flatten[T](items: T) -> T: ...


@overload
def flatten[T](items: typing.Iterable[T | typing.Iterable]) -> list[T]: ...


def flatten[T](items: T | Iterable[T | Iterable]) -> T | list[T]:
    """Flatten a nested list.

    This function takes any nested iterable and returns a flat list.

    Args:
        items (T | typing.Iterable[T | typing.Iterable]): The nested
            iterable to flatten.

    Returns:
        T | list[T]: The flattened list or the original ``items`` value
        if it is not an iterable or is of type :class:`str`.
    """
    if not isinstance(items, Iterable) or isinstance(items, str):
        # Not iterable
        return items

    # Init flat list
    flattened = []

    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, str):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)


def is_path_type(path: Any) -> TypeGuard[FilePath]:
    """Check if an object is a valid path type.

    This function takes any object and checks if it is a valid path
    type. A valid path type is either a :class:`str`, :class:`bytes` or
    :class:`os.PathLike` object.

    Args:
        path: The object to check.

    Returns:
        :data:`True` if the object is a valid path type, :data:`False`
        otherwise.
    """
    return isinstance(path, (str, bytes, os.PathLike))


def is_image_file(path: FilePath) -> bool:
    """Check if a file is an image.

    This function takes a file path and checks if it is an image file.
    This is done by checking if the file exists and if it has a valid
    image extension.

    Args:
        path: The path to the file.

    Returns:
        :data:`True` if the file is an image, :data:`False` otherwise.
    """
    return os.path.isfile(path) and imghdr.what(path) is not None

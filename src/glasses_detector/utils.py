"""
.. class:: FilePath

.. data:: FilePath
    :noindex:
    :type: typing.TypeAliasType
    :value: str | bytes | os.PathLike

    Type alias for a file path.

    Bound:
        :class:`str` | :class:`bytes` | :class:`os.PathLike`
"""
import imghdr
import os
from typing import Any, Iterable, TypeGuard
from urllib.parse import urlparse

type FilePath = str | bytes | os.PathLike


def is_url(x: str) -> bool:
    """Check if a string is a valid URL.

    Takes any string and checks if it is a valid URL. This is taken from
    https://stackoverflow.com/a/38020041.

    Args:
        x (str): The string to check.

    Returns:
        bool: ``True`` if the string is a valid URL, ``False``
        otherwise.
    """
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


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
        path (typing.Any): The object to check.

    Returns:
        typing.TypeGuard[FilePath]: ``True`` if the object is a valid
        path type, ``False`` otherwise.
    """
    return isinstance(path, (str, bytes, os.PathLike))


def is_image_file(path: FilePath) -> bool:
    """Check if a file is an image.

    This function takes a file path and checks if it is an image file.
    This is done by checking if the file exists and if it has a valid
    image extension.

    Args:
        path (FilePath): The path to the file.

    Returns:
        bool: ``True`` if the file is an image, ``False`` otherwise.
    """
    return os.path.isfile(path) and imghdr.what(path) is not None

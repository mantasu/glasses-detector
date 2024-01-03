import imghdr
import os
from typing import Iterable, TypeVar
from urllib.parse import urlparse

T = TypeVar("T")


def is_url(x: str) -> bool:
    # https://stackoverflow.com/a/38020041
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def flatten(items: T | Iterable[T | Iterable]) -> T | list[T]:
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


def is_image(path: os.PathLike) -> bool:
    return os.path.isfile(path) and imghdr.what(path) is not None

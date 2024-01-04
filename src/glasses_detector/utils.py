import imghdr
import os
from typing import Any, Iterable, TypeGuard, TypeVar
from urllib.parse import urlparse

T = TypeVar("T")
type ImgPath = str | bytes | os.PathLike


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


def is_image_path(path: Any) -> TypeGuard[ImgPath]:
    return isinstance(path, (str, bytes, os.PathLike))


def is_image_file(path: ImgPath) -> bool:
    return os.path.isfile(path) and imghdr.what(path) is not None

import json
import os
import pickle
import warnings
from enum import Enum
from typing import Any, Collection, Iterable, Self, TypeGuard, TypeVar
from urllib.parse import urlparse

import numpy as np
import torch
import yaml
from PIL import Image

type Scalar = bool | int | float | str | np.generic | torch.Tensor
type Tensor = Iterable[Scalar | Tensor] | Image.Image
type Default = Scalar | Tensor

type StandardScalar = bool | int | float | str
type StandardTensor = list[StandardScalar | StandardTensor]
type StandardDefault = StandardScalar | StandardTensor

type NonDefault = TypeVar("NonDefault")


class PredType(Enum):
    """Enum class for prediction types.

    This class contains the default prediction types

    Returns:
        _type_: _description_
    """

    type SCALAR = bool | int | float | str | np.generic
    type ARRAY = Collection[SCALAR | ARRAY]
    type IMAGE = Image.Image
    type DEFAULT = SCALAR | ARRAY | IMAGE
    type NON_DEFAULT = TypeVar("NON_DEFAULT")
    type ANY = DEFAULT | NON_DEFAULT

    @classmethod
    def standardize(cls, pred: Default) -> StandardDefault:
        if isinstance(pred, (bool, int, float, str)):
            return pred
        elif isinstance(pred, np.generic):
            return cls.standardize(pred.item())
        elif isinstance(pred, torch.Tensor) and pred.numel() == 1:
            return cls.standardize(pred.item())
        elif isinstance(pred, Image.Image):
            return np.asarray(pred).tolist()
        elif isinstance(pred, Iterable):
            return [cls.standardize(item) for item in pred]
        else:
            raise ValueError(f"Cannot standardize {type(pred)}")

    @classmethod
    def save(cls, pred: DEFAULT | dict[str, DEFAULT], filepath: os.PathLike):
        def _standardize(pred) -> StandardDefault | dict[str, StandardDefault]:
            if isinstance(pred, dict):
                return {k: cls.standardize(v) for k, v in pred.items()}
            else:
                return cls.standardize(pred)

        def _as_numpy(pred) -> Scalar | np.ndarray:
            if cls.is_scalar(pred):
                return pred
            elif isinstance(pred, dict):
                # Stack to single 2D matrix (flatten lists)
                names_col = np.array(pred.keys())[:, None]
                vals_cols = np.stack(
                    [np.atleast_1d(flatten(cls.standardize(v))) for v in pred.values()]
                )
                return np.hstack((names_col, vals_cols))
            else:
                return np.array(cls.standardize(pred))

        # Make the directory to save the file to and get ext
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ext = os.path.splitext(filepath)[1]

        match ext:
            case ".txt":
                if cls.is_scalar(pred):
                    with open(filepath, "w") as f:
                        f.write(str(pred))
                else:
                    # Save to .txt each row has image name and pred values
                    np.savetxt(filepath, _as_numpy(pred), delimiter=" ")
            case ".csv":
                if cls.is_scalar(pred):
                    with open(filepath, "w") as f:
                        f.write(str(pred))
                else:
                    # Save to .csv each row has image name and pred values
                    np.savetxt(filepath, _as_numpy(pred), delimiter=",")
            case ".json":
                with open(filepath, "w") as f:
                    json.dump(_standardize(pred), f)
            case ".yml" | ".yaml":
                with open(filepath, "w") as f:
                    yaml.dump(_standardize(pred), f)
            case ".pkl":
                with open(filepath, "wb") as f:
                    pickle.dump(pred, f)
            case ".npy":
                np.save(filepath, _as_numpy(pred))
            case ".npz":
                np.savez_compressed(filepath, _as_numpy(pred))
            case ".dat":
                if isinstance(pred, Iterable):
                    np.array(cls.standardize(pred)).tofile(filepath)
                else:
                    with open(filepath, "wb") as f:
                        np.savetxt(f, cls.standardize(pred))
            case ".jpg" | ".jpeg" | ".png" | ".bmp" | ".pgm" | ".webp":
                if isinstance(pred, dict) and len(pred) > 1:
                    dirname = os.path.splitext(filepath)[0]
                    os.makedirs(dirname, exist_ok=True)

                    warnings.warn(
                        f"Cannot save multiple images to a single file "
                        f"(prediction type is dict). All images (interpreted "
                        f"from values) will be saved to {dirname} with "
                        f"corresponding file names (interpreted from keys)."
                    )
                else:
                    dirname = os.path.dirname(filepath)
                    pred = {os.path.basename(filepath): pred}

                for name, img in pred.items():
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(np.atleast_1d(cls.standardize(img)))

                    img.save(os.path.join(dirname, name))
            case _:
                raise ValueError(f"Cannot save to {ext} file (not supported).")

    @staticmethod
    def is_scalar(pred: Any) -> TypeGuard[SCALAR]:
        return isinstance(pred, (bool, int, float, str, np.generic))

    @staticmethod
    def is_image(pred: Any) -> TypeGuard[IMAGE]:
        return isinstance(pred, Image.Image)

    @classmethod
    def is_default(cls, pred: Any) -> TypeGuard[DEFAULT]:
        return cls.check(pred) != cls.OTHER

    @classmethod
    def check(cls, pred: Any) -> Self:
        # Get the type of the prediction
        if cls.is_scalar(pred):
            return cls.SCALAR
        elif cls.is_list1d(pred):
            return cls.LIST1D
        elif cls.is_image(pred):
            return cls.IMAGE
        else:
            return cls.NON_DEFAULT


T = TypeVar("T")


def is_url(x):
    # https://stackoverflow.com/a/38020041
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def flatten(items: T | Iterable[T | Iterable]) -> list[T]:
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

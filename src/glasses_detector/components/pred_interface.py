import json
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Collection, Iterable, overload

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from ..utils import ImgPath, flatten, is_image_file
from .pred_type import PredType as PT


class PredInterface(ABC):
    @staticmethod
    def save(pred: PT.Default | dict[str, PT.Default], filepath: ImgPath):
        def _standardize(pred) -> PT.StandardDefault | dict[str, PT.StandardDefault]:
            if isinstance(pred, dict):
                return {k: PT.standardize(v) for k, v in pred.items()}
            else:
                return PT.standardize(pred)

        def _as_numpy(pred) -> PT.Scalar | np.ndarray:
            if PT.is_scalar(pred):
                return pred
            elif isinstance(pred, dict):
                # Stack to single 2D matrix (flatten lists)
                names_col = np.array(pred.keys())[:, None]
                vals_cols = np.stack(
                    [np.atleast_1d(flatten(PT.standardize(v))) for v in pred.values()]
                )
                return np.hstack((names_col, vals_cols))
            else:
                return np.array(PT.standardize(pred))

        # Make the directory to save the file to and get ext
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ext = os.path.splitext(filepath)[1]

        match ext:
            case ".txt":
                if PT.is_scalar(pred):
                    with open(filepath, "w") as f:
                        f.write(str(pred))
                else:
                    # Save to .txt each row has image name and pred values
                    np.savetxt(filepath, _as_numpy(pred), delimiter=" ")
            case ".csv":
                if PT.is_scalar(pred):
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
                    np.array(PT.standardize(pred)).tofile(filepath)
                else:
                    with open(filepath, "wb") as f:
                        np.savetxt(f, PT.standardize(pred))
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
                        img = Image.fromarray(np.atleast_1d(PT.standardize(img)))

                    img.save(os.path.join(dirname, name))
            case _:
                raise ValueError(f"Cannot save to {ext} file (not supported).")

    @abstractmethod
    def predict(
        self,
        image: ImgPath | Collection[ImgPath],
        **kwargs,
    ) -> PT.Default | list[PT.Default]:
        ...

    @overload
    def process_file(
        self,
        input_path: ImgPath,
        output_path: ImgPath | None = None,
        ext: str | None = None,
        show: bool = False,
        **pred_kwargs,
    ) -> PT.Default | None:
        ...

    @overload
    def process_file(
        self,
        input_path: Collection[ImgPath],
        output_path: Collection[ImgPath] | None = None,
        ext: str | None = None,
        show: bool = False,
        **pred_kwargs,
    ) -> list[PT.Default | None]:
        ...

    def process_file(
        self,
        input_path: ImgPath | Collection[ImgPath],
        output_path: ImgPath | Collection[ImgPath] | None = None,
        ext: str | None = None,
        show: bool = False,
        **pred_kwargs,
    ) -> PT.Default | None | list[PT.Default | None]:
        is_multiple = isinstance(input_path, Collection) and not isinstance(
            input_path, str
        )
        input_paths = input_path if is_multiple else [input_path]
        safe_paths = []

        for path in input_paths:
            if not is_image_file(path):
                # Raise a warning if not an image is passed; set to None
                warnings.warn(f"{input_path} is not an image. Skipping...")
            else:
                # Append to safe paths
                safe_paths.append(path)

        try:
            # Predict using the child method by passing the image paths
            preds = self.predict(safe_paths, **pred_kwargs)
            preds = [preds.pop() if p in safe_paths else None for p in input_paths]
        except Exception as e:
            # Raise a warning if prediction failed and return None
            warnings.warn(f"Prediction failed for {input_paths}. Skipping...\n{e}")
            return [None] * len(input_paths) if is_multiple else None

        if show:
            for pred in preds:
                if isinstance(pred, Image.Image):
                    # Show as image
                    plt.show(pred)
                else:
                    # To stdout
                    print(pred)

        if output_path is None:
            # Output is None or a single file for multiple inputs
            output_paths = [None] * len(input_paths)
        elif isinstance(output_path, Collection):
            # Output is a list of paths
            output_paths = output_path
        elif is_multiple and os.path.splitext(output_path)[1] != "":
            # Output is a single file for multiple inputs
            output_paths = [None] * len(input_paths)
        elif is_multiple:
            # Output is a dir for multiple inputs
            output_paths = [output_path] * len(input_paths)
        else:
            # Output is a single file for a single input
            output_paths = [output_path]

        if len(output_path) != len(input_paths):
            warnings.warn(
                f"Number of output paths ({len(output_paths)}) does not match "
                f"the number of input paths ({len(input_paths)}). The number "
                f"of predictions will be truncated or expanded with 'Nones' "
                f"to match the number of input paths."
            )

        while len(output_paths) < len(input_paths):
            # Append None to output paths
            output_paths.append(None)

        for inp, out, pred in zip(input_paths, output_paths, preds):
            if pred is None:
                continue

            if (split_path := os.path.splitext(out))[1] == "":
                if (_ext := ext) is None:
                    # Automatically determine the extension if not given
                    _ext = ".jpg" if isinstance(pred, Image.Image) else ".csv"

                # Output path is dir so input path filename is used
                no_ext = os.path.splitext(os.path.basename(inp))[0]
                out = os.path.join(split_path[0], no_ext + _ext)

            if out is not None:
                # Save pred to file
                self.save(pred, out)

        if (
            is_multiple
            and isinstance(output_path, ImgPath)
            and os.path.splitext(output_path)[1] != ""
        ):
            # Output path is a single file for multiple inputs
            safe_preds = {
                path: pred
                for path, pred in zip(input_paths, preds)
                if path in safe_paths
            }
            self.save(safe_preds, output_path)

        return preds if is_multiple else preds[0]

    def process_dir(
        self,
        input_path: ImgPath,
        output_path: ImgPath | None = None,
        ext: str | None = None,
        batch_size: int = 1,
        show: bool = False,
        pbar: bool | str | tqdm = True,
        update_total: bool = True,
        **pred_kwargs,
    ) -> dict[str, PT.Default | None] | None:
        if isinstance(pbar, bool) and pbar:
            pbar = ""

        if isinstance(pbar, str):
            pbar = tqdm(desc=pbar, total=0, unit="file")
            update_total = True

        if isinstance(pbar, tqdm) and update_total:
            pbar.total += len(os.listdir(input_path))

        # Check if the predictions should be aggregated to
        is_agg = output_path is None or os.path.splitext(output_path)[1] != ""
        pred_dict = {} if is_agg else None

        # Create a list of file batches to process (bs is 1 by default)
        files = [entry.path for entry in os.scandir(input_path) if entry.is_file()]
        files = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

        for input_paths in files:
            if is_agg:
                # Predictions will be aggregated to a single file
                output_paths = None
            elif ext is None:
                # Set output path to a dir (ext will be automatic)
                output_paths = [output_path] * len(input_paths)
            else:
                # Create full output paths by replacing the ext
                output_paths = [
                    os.path.join(
                        output_path,
                        os.path.splitext(os.path.basename(p))[0] + ext,
                    )
                    for p in input_paths
                ]

            # Get the predictions for the batch (some may be None)
            preds = self.process_file(
                input_path=input_paths,
                output_path=output_paths,
                show=show,
                **pred_kwargs,
            )

            if is_agg:
                # Store safe predictions only if aggregation is needed
                pred_dict.update(
                    {
                        os.path.basename(path): pred
                        for path, pred in zip(input_paths, preds)
                        if pred is not None
                    }
                )

            if pbar:
                # Update the progress bar
                pbar.update(len(input_paths))

        if output_path is None:
            return pred_dict

        if is_agg:
            # Save aggregated predictions to a single file
            self.save(pred_dict, output_path)

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

from ..utils import FilePath, flatten, is_image_file
from .pred_type import Default, PredType, Scalar, StandardDefault


class PredInterface(ABC):
    """Interface for handling image-based predictions.

    This interface provides a common API for handling predictions
    (e.g. saving, loading, etc.) for all models. It also provides
    a common API for making predictions on images, directories, and
    files.

    Any class that inherits from this interface must implement the
    :meth:`predict` method. This method should take a single image path
    or a list of image paths and return a single prediction or a list
    of predictions, respectively.
    """

    @staticmethod
    def save(pred: Default | dict[str, Default], filepath: FilePath):
        """Saves a prediction to a file.

        This method saves a prediction or a dictionary of named
        predictions to the provided file. The type of the file will be
        inferred automatically from the extension and will be saved
        accordingly.

        .. list-table:: Supported File Types
            :header-rows: 1

            * - Extension
              - Format
            * - ``.txt``
              - For a single prediction, it is flattened and saved
                as a single line separated by spaces. For a dictionary
                of predictions, each row contains the name of the
                prediction followed by the flattened prediction values
                separated by spaces (no header!).
            * - ``.csv``
              - For a single prediction, it is flattened and saved
                as a single line separated by commas. For a dictionary
                of predictions, each row contains the name of the
                prediction followed by the flattened prediction values
                separated by commas (no header!).
            * - ``.json``
              - For a single prediction, it is saved as a single
                JSON object. For a dictionary of predictions, each
                prediction is saved as a separate JSON object with the
                name of the prediction as the key.
            * - ``.yml``, ``.yaml``
              - For a single prediction, it is saved as
                a single YAML object. For a dictionary of predictions,
                each prediction is saved as a separate YAML object with
                the name of the prediction as the key.
            * - ``.pkl``
              - For a single prediction, it is saved as a single
                pickle object. For a dictionary of predictions, each
                prediction is saved as a separate pickle object with the
                name of the prediction as the key.
            * - ``.npy``, ``.npz``
              - For a single prediction, it is saved as
                a single numpy array or scalar. For a dictionary of
                predictions, it is flattened to a 2D matrix where each
                row contains the name of the prediction followed by the
                flattened prediction values. For ``.npy``,
                :func:`numpy.save` is used and for ``.npz``,
                :func:`numpy.savez_compressed` is used.
            * - ``.dat``
              - For a single prediction, it is saved as a single
                numpy array or scalar using
                :meth:`numpy.ndarray.tofile`. For a dictionary of
                predictions, they are first flattened to a 2D matrix
                before saving.
            * - ``.jpg``, ``.jpeg``, ``.png``, ``.bmp``, ``.pgm``,
                ``.webp``
              - For a single prediction, it is saved as an image. For a
                dictionary of predictions, each prediction is saved as a
                separate image with the name of the prediction as the
                file name. In the case of multiple predictions, all
                images are saved under directory called ``filepath``,
                just without an extension.

        Args:
            pred: The single prediction or a dictionary of predictions
                to save.
            filepath: The path to save the prediction(-s) to.

        Raises:
            ValueError: If the file type is not supported.
        """

        def _standardize(pred) -> StandardDefault | dict[str, StandardDefault]:
            if isinstance(pred, dict):
                return {k: PredType.standardize(v) for k, v in pred.items()}
            else:
                return PredType.standardize(pred)

        def _as_numpy(pred) -> Scalar | np.ndarray:
            if PredType.is_scalar(pred):
                return pred
            elif isinstance(pred, dict):
                # Stack to single 2D matrix (flatten lists)
                names_col = np.array(pred.keys())[:, None]
                vals_cols = np.stack(
                    [
                        np.atleast_1d(flatten(PredType.standardize(v)))
                        for v in pred.values()
                    ]
                )
                return np.hstack((names_col, vals_cols))
            else:
                return np.array(PredType.standardize(pred))

        # Make the directory to save the file to and get ext
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ext = os.path.splitext(filepath)[1]

        match ext:
            case ".txt":
                if PredType.is_scalar(pred):
                    with open(filepath, "w") as f:
                        f.write(str(pred))
                else:
                    # Save to .txt each row has image name and pred values
                    np.savetxt(filepath, _as_numpy(pred), delimiter=" ")
            case ".csv":
                if PredType.is_scalar(pred):
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
                if isinstance(pred, dict):
                    # Stack to 2D matrix
                    pred = _as_numpy(pred)

                if isinstance(pred, Iterable):
                    np.array(PredType.standardize(pred)).tofile(filepath)
                else:
                    with open(filepath, "wb") as f:
                        np.savetxt(f, PredType.standardize(pred))
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
                        img = Image.fromarray(np.atleast_1d(PredType.standardize(img)))

                    img.save(os.path.join(dirname, name))
            case _:
                raise ValueError(f"Cannot save to {ext} file (not supported).")

    @overload
    def predict(
        self,
        image: FilePath,
        **kwargs,
    ) -> Default:
        ...

    @overload
    def predict(
        self,
        image: Collection[FilePath],
        **kwargs,
    ) -> list[Default]:
        ...

    @abstractmethod
    def predict(
        self,
        image: FilePath | Collection[FilePath],
        **kwargs,
    ):
        """Generates a prediction for the given image(-s).

        Takes a path to an image or a list of paths to images and
        returns a prediction or a list of predictions, respectively.

        Args:
            image (FilePath | typing.Collection[FilePath]): The path to
                an image or a list of paths to images to generate
                predictions for.
            **kwargs: Additional keyword arguments to pass to the
                prediction method.

        Returns:
            Default | list[Default]: The prediction or a list of
            predictions for the given image(-s).
        """
        ...

    @overload
    def process_file(
        self,
        input_path: FilePath,
        output_path: FilePath | None = None,
        ext: str | None = None,
        show: bool = False,
        **pred_kwargs,
    ) -> Default | None:
        ...

    @overload
    def process_file(
        self,
        input_path: Collection[FilePath],
        output_path: Collection[FilePath] | None = None,
        ext: str | None = None,
        show: bool = False,
        **pred_kwargs,
    ) -> list[Default | None]:
        ...

    def process_file(
        self,
        input_path: FilePath | Collection[FilePath],
        output_path: FilePath | Collection[FilePath] | None = None,
        ext: str | None = None,
        show: bool = False,
        **pred_kwargs,
    ) -> Default | None | list[Default | None]:
        """Processes a single image or a list of images.

        Takes a path to the image or a list of paths to images,
        generates the prediction(-s), and returns them, based on how
        :meth:`.predict` behaves. If the output path is specified, the
        prediction(-s) will be saved to the given path(-s) based on the
        extension of the output path. The following cases are
        considered:

        1. If ``output_path`` is :data:`None`, no predictions are saved.
           If there are multiple output paths (one for each input path)
           and some of the entries are :data:`None`, then only the
           outputs for the corresponding predictions are not be saved.
        2. If the output path is a single file, then the predictions are
           saved to that file. If there are multiple input paths, then
           the corresponding predictions are aggregated to a single
           file.
        3. If ``output_path`` is a directory, then the prediction(-s)
           are saved to that directory. For each input path, a
           corresponding file is created in the specified output
           directory with the same name as the input. The extension, if
           not provided as ``ext``, is set to ``.jpg`` for images and
           ``.txt`` for other predictions.
        4. If ``output_path`` is a list of output paths, then the
           predictions are saved to the corresponding output paths. If
           the number of input paths and output paths do not match, then
           the number of predictions are be truncated or expanded with
           :data:`None` to match the number of input paths and a warning
           is raised. all the output paths are interpreted as files.

        For more details on how each file type is saved, regardless if
        it is a single prediction or the aggregated predictions, see
        :meth:`.save`

        Tip:
            If multiple images are provided (as a list of input paths),
            they are likely to be loaded into a single batch for a
            faster prediction (see :meth:`.predict` for more details),
            thus more memory is required than if they were processed
            individually. For this reason, consider not to pass too many
            images at once (e.g., <200).

        Note:
            If some input path does not lead to a valid image file,
            e.g., does not exist, its prediction is set to :data:`None`.
            Also, if at least one prediction fails, then a all
            predictions are set to :data:`None`. In both cases, a
            warning is is raised and the files or the lines in the
            aggregated file are skipped (not saved).

        Args:
            input_path (FilePath | typing.Collection[FilePath]): The
                path to an image or a list of paths to images to
                generate predictions for.
            output_path (FilePath | typing.Collection[FilePath] | None, optional):
                The path to save the prediction(-s) to. If :data:`None`,
                no predictions are saved. If a single file, the
                predictions are aggregated (if multiple) and saved to
                that file. If a directory, the predictions are saved to
                that directory with the names copied from inputs.
                Defaults to :data:`None`.
            ext (str | None, optional): The extension to use for the
                output file(-s). Only used when ``output_path`` is a
                directory. If :data:`None`, the extension is set to
                ``".jpg"`` for images and ``".txt"`` for other
                predictions (depends on what is returned by
                :meth:`.predict` returns) For available options, refer
                to :meth:`.save`. Defaults to :data:`None`.
            show (bool, optional): Whether to show the predictions.
                Images will be shown using
                :func:`matplotlib.pyplot.show` and other predictions
                will be printed to stdout. Defaults to :data:`False`.
            **pred_kwargs: Additional keyword arguments to pass to
                :meth:`.predict`.

        Returns:
            Default | None | list[Default | None]: The prediction or a
            list of predictions for the given image(-s). Any failed
            predictions will be set to :data:`None`.
        """
        is_multiple = not isinstance(input_path, str) and isinstance(
            input_path, Collection
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
        elif not isinstance(output_path, str) and isinstance(output_path, Collection):
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
                    _ext = ".jpg" if isinstance(pred, Image.Image) else ".txt"

                # Output path is dir so input path filename is used
                no_ext = os.path.splitext(os.path.basename(inp))[0]
                out = os.path.join(split_path[0], no_ext + _ext)

            if out is not None:
                # Save pred to file
                self.save(pred, out)

        if (
            is_multiple
            and isinstance(output_path, FilePath)
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
        input_path: FilePath,
        output_path: FilePath | None = None,
        ext: str | None = None,
        batch_size: int = 1,
        show: bool = False,
        pbar: bool | str | tqdm = True,
        update_total: bool = True,
        **pred_kwargs,
    ) -> dict[str, Default | None] | None:
        """Processes a directory of images.

        Takes a path to a directory of images, optionally sub-groups to
        batches, generates the predictions for every image and returns
        them if ``output_path`` is :data:`None` or saves them to a specified
        file or as files to a specified directory. The following cases
        are considered:

        1. If ``output_path`` is :data:`None`, the predictions are returned
           as a dictionary of predictions where the keys are the names
           of the images and the values are the corresponding
           predictions.
        2. If ``output_path`` is a single file, the predictions are
           aggregated to a single file.
        3. If ``output_path`` is a directory, the predictions are saved
           to that directory. For each input path, a corresponding file
           is created in the specified output directory with the same
           name as the input. The extension, if not provided as ``ext``,
           is set automatically as explained in :meth:`.process_file`.

        For more details on how each file type is saved, regardless if
        it is a single prediction or the aggregated predictions, see
        :meth:`.save`.

        Tip:
            For *very large* directories, consider specifying
            ``output_path`` as a directory because aggregating the
            predictions to a single file or waiting for them to be
            returned might consume too much memory and lead to errors.

        Note:
            Any files in the input directory that are not valid images
            or those for which the prediction fails for any reason are
            are simply skipped and a warning is raised - for more
            details, see :meth:`.process_file`.

        Args:
            input_path (FilePath): The path to a directory of images to
                generate predictions for.
            output_path (FilePath | None, optional): The path to save the
                prediction(-s) to. If :data:`None`, the predictions are
                returned as a dictionary, if a single file, the
                predictions are aggregated to a single file, and if a
                directory, the predictions are saved to that directory
                with the names copied from inputs. Defaults to
                :data:`None`.
            ext (str | None, optional): The extension to use for the
                output file(-s). Only used when ``output_path`` is a
                directory. The extension should include a leading dot,
                e.g., ``".txt"``, ``".npy"``, ``".jpg"`` etc
                (see :meth:`save`). If :data:`None`, the behavior
                follows :meth:`.process_file`. Defaults to :data:`None`.
            batch_size (int, optional): The batch size to use when
                processing the images. This groups the files in the
                specified directory to batches of size ``batch_size``
                before processing them. In some cases, larger batch
                sizes can speed up the processing at the cost of more
                memory usage. Defaults to ``1``.
            show (bool, optional): Whether to show the predictions.
                Images will be shown using
                :func:`matplotlib.pyplot.show` and other predictions
                will be printed to stdout. It is not recommended to set
                this to :data:`True` as it might spam your stdout.
                Defaults to :data:`False`.
            pbar (bool | str | tqdm.tqdm, optional): Whether to show a
                progress bar. If :data:`True`, a progress bar with no
                description is shown. If :class:`str`, a progress bar
                with the given description is shown. If an instance of
                :class:`~tqdm.tqdm`, it is used as is. Defaults to
                :data:`True`.
            update_total (bool, optional): Whether to update the total
                number of files in the progress bar. This is only
                relevant if ``pbar`` is an instance of
                :class:`~tqdm.tqdm`. For example, if the number of total
                files is already known and captured by
                :attr:`tqdm.tqdm.total`, then there is no need to update
                it. Defaults to :data:`True`.
            **pred_kwargs: Additional keyword arguments to pass to
                :meth:`.predict`.

        Returns:
            dict[str, Default | None] | None: The dictionary of
            predictions if ``output_path`` is :data:`None` or
            :data:`None` if ``output_path`` is specified.
        """
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

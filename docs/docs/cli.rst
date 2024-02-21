:fas:`terminal` CLI
===================

.. role:: bash(code)
  :language: bash
  :class: highlight

These flags allow you to define the kind of task and the model to process your image or a directory with images. Check out how to use them in :ref:`command-line`.

.. option:: -i path/to/dir/or/file, --input path/to/dir/or/file

    Path to the input image or the directory with images.

.. option:: -o path/to/dir/or/file, --output path/to/dir/or/file

    Path to the output file or the directory. If not provided, then, if input is a file, the prediction will be printed (or shown if it is an image), otherwise, if input is a directory, the predictions will be written to a directory with the same name with an added suffix ``_preds``. If provided as a file, then the prediction(-s) will be saved to this file (supported extensions include: ``.txt``, ``.csv``, ``.json``, ``.npy``, ``.pkl``, ``.jpg``, ``.png``). If provided as a directory, then the predictions will be saved to this directory use :bash:`--extension` flag to specify the file extensions in that directory.
    
    **Default:** :py:data:`None`

.. option:: -e <ext>, --extension <ext>

    Only used if :bash:`--output` is a directory. The extension to use to save the predictions as files. Common extensions include: ``.txt``, ``.csv``, ``.json``, ``.npy``, ``.pkl``, ``.jpg``, ``.png``. If not specified, it will be set automatically to ``.jpg`` for image predictions and to ``.txt`` for all other formats.
    
    **Default:** :py:data:`None`

.. option:: -f <format>, --format <format>

    The format to use to map the raw prediction to.

    * For *classification*, common formats are ``bool``, ``proba``, ``str`` - check :meth:`GlassesClassifier.predict<glasses_detector.classifier.GlassesClassifier.predict>` for more details
    * For *detection*, common formats are ``bool``, ``int``, ``img`` - check :meth:`GlassesDetector.predict<glasses_detector.detector.GlassesDetector.predict>` for more details
    * For *segmentation*, common formats are ``proba``, ``img``, ``mask`` - check :meth:`GlassesSegmenter.predict<glasses_detector.segmenter.GlassesSegmenter.predict>` for more details

    If not specified, it will be set automatically to ``str``, ``img``, ``mask`` for *classification*, *detection*, *segmentation* respectively.
    
    **Default:** :py:data:`None`

.. option:: -t <task-name>, --task <task-name>

    The kind of task the model should perform. One of

    * ``classification``
    * ``classification:anyglasses``
    * ``classification:sunglasses``
    * ``classification:eyeglasses``
    * ``detection``
    * ``detection:eyes``
    * ``detection:solo``
    * ``detection:worn``
    * ``segmentation``
    * ``segmentation:frames``
    * ``segmentation:full``
    * ``segmentation:legs``
    * ``segmentation:lenses``
    * ``segmentation:shadows``
    * ``segmentation:smart``

    If specified only as ``classification``, ``detection``, or ``segmentation``, the subcategories ``anyglasses``, ``worn``, and ``smart`` will be chosen, respectively.

    **Default:** ``classification:anyglasses``

.. option:: -s <model-size>, --size <model-size>

    The model size which determines architecture type. One of ``small``, ``medium``, ``large``.
    
    **Default:** ``medium``

.. option:: -b <batch-size>, --batch-size <batch-size>

    Only used if :bash:`--input` is a directory. The batch size to use when processing the images. This groups the files in the input directory to batches of size ``batch_size`` before processing them. In some cases, larger batch sizes can speed up the processing at the cost of more memory usage.
    
    **Default:** ``1``

.. option:: -p <pbar-desc>, --pbar <pbar-desc>

    Only used if :bash:`--input` is a directory. It is the description that is used for the progress bar. If specified as ``""`` (empty string), no progress bar is shown.
    
    **Default:** ``"Processing"``

.. option:: -w path/to/weights.pth, --weights path/to/weights.pth

    Path to custom weights to load into the model. If not specified, weights will be loaded from the default location (and automatically downloaded there if needed).
    
    **Default:** :py:data:`None`

.. option:: -d <device>, --device <device>

    The device on which to perform inference. If not specified, it will be automatically checked if `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ or `MPS <https://developer.apple.com/documentation/metalperformanceshaders>`_ is supported.
    
    **Default:** :py:data:`None`

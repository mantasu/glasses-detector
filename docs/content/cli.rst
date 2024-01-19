CLI
===

These flags allow you to define the kind of task and the model to process your image or a directory with images. Check out how to use them in :ref:`command-line`.

.. option:: -i <path/to/dir_or_file>, --input-path <path/to/dir_or_file>

    Path to the input image or the directory with images.

.. option:: -o <path/to/dir_or_file>, --output-path <path/to/dir_or_file>

    For classification, it is the path to a file, e.g., txt or csv, to which to write the predictions. If not provided, the prediction will be either printed (if input is a file) or written to a default file (if input is a dir). For segmentation, it is a path to a mask file, e.g., jpg or png, (if input is a file) or a path to a directory where the masks should be saved (if input is a dir). If not provided, default output paths will be generated.
    
    **Default:** ``None``

.. option:: -k <kind-of-model>, --kind <kind-of-model>

    The kind of model to use to process the files. One of 'eyeglasses-classifier', 'sunglasses-classifier', 'glasses-classifier', 'full-glasses-segmenter', 'full-eyeglasses-segmenter', 'full-sunglasses-segmenter', 'full-anyglasses-segmenter', 'glass-frames-segmenter', 'eyeglasses-frames-segmenter', 'sunglasses-frames-segmenter', 'anyglasses-frames-segmenter'.

.. option:: -s <arch-name>, --size <arch-name>

    The model architecture name (model size). One of 'tiny', 'small', 'medium', 'large', 'huge'.
    
    **Default:** ``'small'``

.. option:: -l <label-type>, --label-type <label-type>

    Only used if ``kind`` is classifier. It is the string specifying the way to map the predictions to labels. For instance, if specified as 'int', positive labels will be 1 and negative will be 0. If specified as 'proba', probabilities of being positive will be shown. One of 'bool', 'int', 'str', 'logit', 'proba'. 

    **Default:** ``'int'``

.. option:: -sep <sep>, --separator <sep>

    Only used if ``kind`` is classifier. It is the separator to use to separate image file names and the predictions.
    
    **Default:** ``','``

.. option:: -m <mask-type>, --mask-type <mask-type>

    Only used if ``kind`` is segmenter. The type of mask to generate. For example, a mask could be a black and white image, in which case 'img' should be specified. A mask could be a matrix of raw scores in npy format, in which case 'logit' should be specified. One of 'bool', 'int', 'img', 'logit', 'proba'.
    
    **Default:** ``'img'``

.. option:: -ext <ext>, --extension <ext>

    Only used if ``kind`` is segmenter. The extension to use to save masks. Specifying it will overwrite the extension existing as part of ``output_path`` (if it is specified as a path to file). If ``mask-type`` is 'img', then possible extensions are 'jpg', 'png', 'bmp' etc. If ``mask-type`` is some value, e.g., 'bool' or 'proba', then possible extensions are 'npy', 'pkl', 'dat' etc. If not specified, it will be inferred form ``output-path`` (if it is given and is a path to a file), otherwise 'jpg' or 'npy' will be used, depending on ``mask-type``.
    
    **Default:** ``None``

.. option:: -pbd <pbar-desc> --pbar-desc <pbar-desc>

    Only used if input path leads to a directory of images. It is the description that is used for the progress bar. If specified as ``''`` (empty string), no progress bar is shown.
    
    **Default:** ``'Processing'``

.. option:: -d <device> --device <device>

    The device on which to perform inference. If not specified, it will be automatically checked if CUDA or MPS is supported.
    
    **Default:** ``''``

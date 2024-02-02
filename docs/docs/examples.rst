:fas:`code` Examples
====================

.. role:: bash(code)
  :language: bash
  :class: highlight

.. _command-line:

Command Line
------------

You can run predictions via the command line. For example, classification of a single or multiple images, can be performed via:

.. code-block:: bash

    glasses-detector -i path/to/img.jpg --task classification   # Prints "present" or "absent"
    glasses-detector -i path/to/dir --output path/to/output.csv # Creates CSV (default --task is classification)


It is possible to specify the **kind** of :bash:`--task` in the following format :bash:`task:kind`, for example, we may want to classify only *sunglasses* (only glasses with opaque lenses). Further, more options can be specified, like :bash:`--format`, :bash:`--size`, :bash:`--batch-size`, :bash:`--device`, etc:

.. code-block:: bash
    
    glasses-detector -i path/to/img.jpg -t classification:sunglasses -f proba # Prints probability of sunglasses
    glasses-detector -i path/to/dir -o preds.pkl -s large -b 64 -d cuda       # Fast and accurate processing

Running *detection* and *segmentation* is similar, though we may want to generate a folder of predictions when processing a directory (but we can also squeeze all the predictions into a single file, such as ``.npy``):

.. code-block:: bash

    glasses-detector -i path/to/img.jpg -t detection                          # Shows image with bounding boxes
    glasses-detector -i path/to/dir -t segmentation -f mask -e .jpg           # Generates dir with masks

.. tip::

    For a more exhaustive explanation of the available options use :bash:`glasses-detector --help` or check :doc:`cli`.


Python Script
-------------

The most straightforward way to perform a prediction on a single file (or a list of files) is to use :meth:`~glasses_detector.components.pred_interface.PredInterface.process_file`. Although the prediction(-s) can be saved to a file or a directory, in most cases, this is useful to immediately show the prediction result(-s).

.. code-block:: python
    :linenos:

    from glasses_detector import GlassesClassifier, GlassesDetector

    # Prints either '1' or '0'
    classifier = GlassesClassifier()
    classifier.process_file(
        input_path="path/to/img.jpg",     # can be a list of paths
        format={True: "1", False: "0"},   # similar to format="int"
        show=True,                        # to print the prediction
    )

    # Opens a plot in a new window
    detector = GlassesDetector()
    detector.process_file(
        image="path/to/img.jpg",          # can be a list of paths
        format="img",                     # to return the image with drawn bboxes
        show=True,                        # to show the image using matplotlib
    )

A more useful method is :meth:`~glasses_detector.components.pred_interface.PredInterface.process_dir` which goes through all the images in the directory and generates the predictions into a single file or a directory of files. Also note how we can specify task ``kind`` and model ``size``:

.. code-block:: python
    :linenos:

    from glasses_detector import GlassesClassifier, GlassesSegmenter

    # Generates a CSV file with image paths and labels
    classifier = GlassesClassifier(kind="sunglasses")
    classifier.process_dir(
        input_path="path/to/dir",         # failed files will raise a warning
        output_path="path/to/output.csv", # path/to/dir/img1.jpg,<pred>...
        format="proba",                   # <pred> is a probability of sunglasses
        pbar="Processing",                # Set to None to disable
    )

    # Generates a directory with masks
    segmenter = GlassesSegmenter(size="large", device="cuda")
    segmenter.process_dir(
        input_path="path/to/dir",         # output dir defaults to path/to/dir_preds
        ext=".jpg",                       # saves each mask in JPG format
        format="mask",                    # output type will be a grayscale PIL image
        batch_size=32,                    # to speed up the processing
        output_size=(512, 512),           # Set to None to keep the same size as image
    )


It is also possible to directly use :meth:`~glasses_detector.components.pred_interface.PredInterface.predict` which allows to process already loaded images. This is useful when you want to incorporate the prediction into a custom pipeline.

.. code-block:: python
    :linenos:

    import numpy as np
    from glasses_detector import GlassesDetector

    # Predicts normalized bounding boxes
    detector = GlassesDetector()
    predictions = detector(
        image=np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        format="float",
    )
    print(type(prediction), len(prediction))  # <class 'list'> 10


.. admonition:: Refer to API documentation for model-specific examples
    
    * :class:`~glasses_detector.classifier.GlassesClassifier` and its :meth:`~glasses_detector.classifier.GlassesClassifier.predict`
    * :class:`~glasses_detector.detector.GlassesDetector` and its :meth:`~glasses_detector.detector.GlassesDetector.predict`
    * :class:`~glasses_detector.segmenter.GlassesSegmenter` and its :meth:`~glasses_detector.segmenter.GlassesSegmenter.predict`

Demo
----

Feel free to play around with some `demo image files <https://github.com/mantasu/glasses-detector/data/demo/>`_. For example, after installing through `pip <https://pypi.org/project/glasses-detector/>`_, you can run:

.. code-block:: bash

    git clone https://github.com/mantasu/glasses-detector && cd glasses-detector/data
    glasses-detector -i demo -o demo_labels.csv --task classification:sunglasses -f proba
    glasses-detector -i demo -o demo_masks -t segmentation:full -f img -e .jpg

Alternatively, you can check out the `demo notebook <https://github.com/mantasu/glasses-detector/notebooks/demo.ipynb>`_ which can be also accessed on `Google Colab <https://colab.research.google.com/github/mantasu/glasses-detector/blob/master/notebooks/demo.ipynb>`_.

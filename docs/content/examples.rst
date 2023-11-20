Examples
========

.. _command-line:

Command Line
------------

You can run predictions via the command line. For example, classification of a single or multiple images, can be performed via:

.. code-block:: bash

    glasses-detector -i path/to/img --kind sunglasses-classifier # Prints 1 or 0
    glasses-detector -i path/to/dir --kind sunglasses-classifier # Generates CSV


Running segmentation is similar, just change the ``kind`` argument:

.. code-block:: bash

    glasses-detector -i path/to/img -k glasses-segmenter # Generates img_mask file
    glasses-detector -i path/to/dir -k glasses-segmenter # Generates dir with masks


.. note:: 
    You can also specify things like ``--output-path``, ``--label-type``, ``--size``, ``--device`` etc. Use ``--glasses-detector -h`` for more details or check the :doc:`cli`.


Python Script
-------------

You can import the package and its models via the python script for more flexibility. Here is an example of how to classify people wearing sunglasses (will generate an output file where each line will contain the name of the image and the predicted label, e.g., `some_image.jpg,1`):

.. code-block:: python

    from glasses_detector import SunglassesClassifier

    classifier = SunglassesClassifier(base_model="small", pretrained=True).eval()

    classifier.predict(
        input_path="path/to/dir", 
        output_path="path/to/output.csv",
        label_type="int",
    )


Using a segmenter is similar, here is an example of using a sunglasses segmentation model:

.. code-block:: python

    from glasses_detector import FullSunglassesSegmenter

    # base_model can also be a tuple: (classifier size, base glasses segmenter size)
    segmenter = FullSunglassesSegmenter(base_model="small", pretrained=True).eval()

    segmenter.predict(
        input_path="path/to/dir",
        output_path="path/to/dir_masks",
        mask_type="img",
    )


.. note:: 
    There is much more flexibility that you can do with the given models, for instance, you can use only base segmenters without accompanying classifiers, or you can define your own prediction methods without resizing images to ``256x256`` (as what is done in the background). For more details refer to :doc:`../modules/modules`, for instance at how segmenter :meth:`~.glasses_detector.bases.base_segmenter.BaseSegmenter.predict` works.


Demo
----

Feel free to play around with some `demo image files <https://github.com/mantasu/glasses-detector/demo/>`_. For example, after installing through `pip <https://pypi.org/project/glasses-detector/>`_, you can run:

.. code-block:: bash

    git clone https://github.com/mantasu/glasses-detector && cd glasses-detector/data
    glasses-detector -i demo -o demo_labels.csv --kind sunglasses-classifier --label str

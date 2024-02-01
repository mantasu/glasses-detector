:fas:`gears` Features
=====================

The following *tasks* are supported:

* **Classification** - binary classification of the presence of glasses and their types.
* **Detection** - binary detection of worn/standalone glasses and eye area.
* **Segmentation** - binary segmentation of glasses and their parts.

Each :attr:`task` has multiple :attr:`kinds` (task categories) and model :attr:`sizes` (architectures with pre-trained weights).

Classification
--------------

+----------------+-------------------------------------+-------------------------------------------+
| **Kind**       | **Description**                     | **Example**                               |
+================+=====================================+===========================================+
| ``anyglasses`` | Any kind glasses/googles/spectacles | .. image:: ../../assets/no_glasses.jpg    |
+----------------+-------------------------------------+-------------------------------------------+
| ``eyeglasses`` | Transparent eyeglasses              | .. image:: ../../assets/eyeglasses.jpg    |
+----------------+-------------------------------------+-------------------------------------------+
| ``sunglasses`` | Opaque and semi-transparent glasses | .. image:: ../../assets/sunglasses.jpg    |
+----------------+-------------------------------------+-------------------------------------------+

.. admonition:: Check classifier performances
    :class: tip

    * `Performance Information of the Pre-trained Classifiers <../modules/glasses_detector.classifier.html#performance-of-the-pre-trained-classifiers>`_: performance of each :attr:`kind`.
    * `Size Information of the Pre-trained Classifiers <../modules/glasses_detector.classifier.html#size-information-of-the-pre-trained-classifiers>`_: efficiency of each :attr:`size`.

Detection
---------

+------------------+-------------------------------------+----------------------------------------+
| **Kind**         | **Description**                     | **Example**                            |
+==================+=====================================+========================================+
| ``"eyes"``       | No glasses, just the eye area       |                                        |
+------------------+-------------------------------------+----------------------------------------+
| ``"standalone"`` | Any glasses in the wild             |                                        |
+------------------+-------------------------------------+----------------------------------------+
| ``"worn"``       | Any glasses that are worn by people |                                        |
+------------------+-------------------------------------+----------------------------------------+

.. admonition:: Check detector performances
    :class: tip

    * `Performance Information of the Pre-trained Detectors <../modules/glasses_detector.detector.html#performance-of-the-pre-trained-detectors>`_: performance of each :attr:`kind`.
    * `Size Information of the Pre-trained Detectors <../modules/glasses_detector.detector.html#size-information-of-the-pre-trained-detectors>`_: efficiency of each :attr:`size`.

Segmentation
------------

+-------------+---------------------------------------------------------------------+------------------------------------------------+
| **Kind**    | **Description**                                                     | **Example**                                    |
+=============+=====================================================================+================================================+
| ``frames``  | Frames (including legs) of any glasses                              |                                                |
+-------------+---------------------------------------------------------------------+------------------------------------------------+
| ``full``    | Frames (including legs) and lenses of any glasses                   | .. image:: ../../assets/eyeglasses_mask.jpg    |
+-------------+---------------------------------------------------------------------+------------------------------------------------+
| ``legs``    | Legs of any glasses                                                 |                                                |
+-------------+---------------------------------------------------------------------+------------------------------------------------+
| ``lenses``  | Lenses of any glasses                                               |                                                |
+-------------+---------------------------------------------------------------------+------------------------------------------------+
| ``shadows`` | Cast shadows on the skin of glasses frames only                     |                                                |
+-------------+---------------------------------------------------------------------+------------------------------------------------+
| ``smart``   | Like ``full`` but does not segment lenses if they are transparent   |                                                |
+-------------+---------------------------------------------------------------------+------------------------------------------------+

.. admonition:: Check segmenter performances
    :class: tip

    * `Performance Information of the Pre-trained Segmenters <../modules/glasses_detector.segmenter.html#performance-of-the-pre-trained-segmenters>`_: performance of each :attr:`kind`.
    * `Size Information of the Pre-trained Segmenters <../modules/glasses_detector.segmenter.html#size-information-of-the-pre-trained-segmenters>`_: efficiency of each :attr:`size`.
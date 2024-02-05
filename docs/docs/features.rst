:fas:`gears` Features
=====================

The following *tasks* are supported:

* **Classification** - binary classification of the presence of glasses and their types.
* **Detection** - binary detection of worn/standalone glasses and eye area.
* **Segmentation** - binary segmentation of glasses and their parts.

Each :attr:`task` has multiple :attr:`kinds` (task categories) and model :attr:`sizes` (architectures with pre-trained weights).

Classification
--------------

.. table:: Classification Kinds
    :widths: 15 31 18 18 18
    :name: classification-kinds

    +----------------+-------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | **Kind**       | **Description**                     | **Examples**                                                                                                                                                                            |
    +================+=====================================+=============================================================+=============================================================+=============================================================+
    | ``anyglasses`` | Identifies any kind of glasses,     | .. image:: ../_static/img/classification-eyeglasses-pos.jpg | .. image:: ../_static/img/classification-sunglasses-pos.jpg | .. image:: ../_static/img/classification-no-glasses-neg.jpg |
    |                | googles, or spectacles.             +-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
    |                |                                     | .. centered:: Positive                                      | .. centered:: Positive                                      | .. centered:: Negative                                      |
    +----------------+-------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
    | ``eyeglasses`` | Identifies only transparent glasses | .. image:: ../_static/img/classification-eyeglasses-pos.jpg | .. image:: ../_static/img/classification-sunglasses-neg.jpg | .. image:: ../_static/img/classification-no-glasses-neg.jpg |
    |                | (here referred as *eyeglasses*)     +-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
    |                |                                     | .. centered:: Positive                                      | .. centered:: Negative                                      | .. centered:: Negative                                      |
    +----------------+-------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
    | ``sunglasses`` | Identifies only opaque and          | .. image:: ../_static/img/classification-eyeglasses-neg.jpg | .. image:: ../_static/img/classification-sunglasses-pos.jpg | .. image:: ../_static/img/classification-no-glasses-neg.jpg |
    |                | semi-transparent glasses (here      +-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
    |                | referred as *sunglasses*)           | .. centered:: Negative                                      | .. centered:: Positive                                      | .. centered:: Negative                                      |
    +----------------+-------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

.. admonition:: Check classifier performances
    :class: tip

    * `Performance Information of the Pre-trained Classifiers <api/glasses_detector.classifier.html#performance-of-the-pre-trained-classifiers>`_: performance of each :attr:`kind`.
    * `Size Information of the Pre-trained Classifiers <api/glasses_detector.classifier.html#size-information-of-the-pre-trained-classifiers>`_: efficiency of each :attr:`size`.

Detection
---------

.. table:: Detection Kinds
    :widths: 15 31 18 18 18
    :name: detection-kinds

    +----------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | **Kind** | **Description**                                                                      | **Examples**                                                                                                                                     |
    +==========+======================================================================================+================================================+================================================+================================================+
    | ``eyes`` | Detects only the eye region, no glasses.                                             | .. image:: ../_static/img/detection-eyes-0.jpg | .. image:: ../_static/img/detection-eyes-1.jpg | .. image:: ../_static/img/detection-eyes-2.jpg |
    +----------+--------------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------+------------------------------------------------+
    | ``solo`` | Detects any glasses in the wild, i.e., standalone glasses that are placed somewhere. | .. image:: ../_static/img/detection-solo-0.jpg | .. image:: ../_static/img/detection-solo-1.jpg | .. image:: ../_static/img/detection-solo-2.jpg |
    +----------+--------------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------+------------------------------------------------+
    | ``worn`` | Detects any glasses worn by people but can also detect non-worn glasses.             | .. image:: ../_static/img/detection-worn-0.jpg | .. image:: ../_static/img/detection-worn-1.jpg | .. image:: ../_static/img/detection-worn-2.jpg |
    +----------+--------------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------+------------------------------------------------+

.. admonition:: Check detector performances
    :class: tip

    * `Performance Information of the Pre-trained Detectors <api/glasses_detector.detector.html#performance-of-the-pre-trained-detectors>`_: performance of each :attr:`kind`.
    * `Size Information of the Pre-trained Detectors <api/glasses_detector.detector.html#size-information-of-the-pre-trained-detectors>`_: efficiency of each :attr:`size`.

Segmentation
------------

.. table:: Segmentation Kinds
    :widths: 15 31 18 18 18
    :name: segmentation-kinds

    +-------------+----------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | **Kind**    | **Description**                                                                                    | **Examples**                                                                                                                                                       |
    +=============+====================================================================================================+======================================================+======================================================+======================================================+
    | ``frames``  | Segments frames (including legs) of any glasses                                                    | .. image:: ../_static/img/segmentation-frames-0.jpg  | .. image:: ../_static/img/segmentation-frames-1.jpg  | .. image:: ../_static/img/segmentation-frames-2.jpg  |
    +-------------+----------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+
    | ``full``    | Segments full glasses, i.e., lenses and the whole frame                                            | .. image:: ../_static/img/segmentation-full-0.jpg    | .. image:: ../_static/img/segmentation-full-1.jpg    | .. image:: ../_static/img/segmentation-full-2.jpg    |
    +-------------+----------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+
    | ``legs``    | Segments only frame legs of standalone glasses                                                     | .. image:: ../_static/img/segmentation-legs-0.jpg    | .. image:: ../_static/img/segmentation-legs-1.jpg    | .. image:: ../_static/img/segmentation-legs-2.jpg    |
    +-------------+----------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+
    | ``lenses``  | Segments lenses of any glasses (both transparent and opaque).                                      | .. image:: ../_static/img/segmentation-lenses-0.jpg  | .. image:: ../_static/img/segmentation-lenses-1.jpg  | .. image:: ../_static/img/segmentation-lenses-2.jpg  |
    +-------------+----------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+
    | ``shadows`` | Segments cast shadows on the skin by the glasses frames only (does not consider opaque lenses).    | .. image:: ../_static/img/segmentation-shadows-0.jpg | .. image:: ../_static/img/segmentation-shadows-1.jpg | .. image:: ../_static/img/segmentation-shadows-2.jpg |
    +-------------+----------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+
    | ``smart``   | Segments visible glasses parts: like ``full`` but does not segment lenses if they are transparent. | .. image:: ../_static/img/segmentation-smart-0.jpg   | .. image:: ../_static/img/segmentation-smart-1.jpg   | .. image:: ../_static/img/segmentation-smart-2.jpg   |
    +-------------+----------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+------------------------------------------------------+

.. admonition:: Check segmenter performances
    :class: tip

    * `Performance Information of the Pre-trained Segmenters <api/glasses_detector.segmenter.html#performance-of-the-pre-trained-segmenters>`_: performance of each :attr:`kind`.
    * `Size Information of the Pre-trained Segmenters <api/glasses_detector.segmenter.html#size-information-of-the-pre-trained-segmenters>`_: efficiency of each :attr:`size`.
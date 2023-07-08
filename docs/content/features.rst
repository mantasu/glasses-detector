Features
========

There are 2 kinds of classifiers and 2 kinds of segmenters (terminology is a bit off but easier to handle with unique names):

* **Eyeglasses classifier** - identifies only transparent glasses, i.e., prescription spectacles.
* **Sunglasses classifier** - identifies only occluded glasses, i.e., sunglasses.
* **Full glasses segmenter** - segments full glasses, i.e., their frames and actual glasses (regardless of the glasses type).
* **Glasses frames segmenter** - segments glasses frames (regardless of the glasses type).

Each kind has 5 different model architectures with naming conventions set from *tiny* to *huge*.

Classification
--------------

A classifier only identifies whether a corresponding category of glasses (transparent eyeglasses or occluded sunglasses) is present:

.. list-table::
    :header-rows: 1

    * - Model type
      - .. image:: ../../assets/eyeglasses.jpg
      - .. image:: ../../assets/sunglasses.jpg
      - .. image:: ../../assets/no_glasses.jpg
    * - Eyeglasses classifier
      - wears
      - doesn't wear
      - doesn't wear
    * - Sunglasses classifier  
      - doesn't wear
      - wears
      - doesn't wear
    * - Any glasses classifier
      - wears
      - wears
      - doesn't wear

These are the performances of *eyeglasses* and *sunglasses* models and their sizes. Note that the joint *glasses* classifier would have an average accuracy and a combined model size of both *eyeglasses* and *sunglasses* models.

.. collapse:: Eyeglasses classification models (performance & weights)

    .. list-table::
        :header-rows: 1

        * - Model type
          - BCE loss :math:`\downarrow`
          - F1 score :math:`\uparrow`
          - ROC-AUC score :math:`\uparrow`
          - Num params :math:`\downarrow`
          - Model size :math:`\downarrow`
        * - *tiny*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *small*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *medium*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *large* 
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA  
        * - *huge*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA

.. collapse:: Sunglasses classification models (performance & weights)

    .. list-table::
        :header-rows: 1

        * - Model type
          - BCE loss :math:`\downarrow`
          - F1 score :math:`\uparrow`
          - ROC-AUC score :math:`\uparrow`
          - Num params :math:`\downarrow`
          - Model size :math:`\downarrow`
        * - *tiny*
          - 0.1149
          - 0.9137
          - 0.9967
          - **27.53 k**
          - **0.11 Mb**
        * - *small*
          - 0.0645
          - 0.9434
          - 0.9987
          - 342.82 k
          - 1.34 Mb
        * - *medium*
          - **0.0491**
          - 0.9651
          - **0.9992**
          - 1.52 M
          - 5.84 Mb
        * - *large*
          - 0.0532
          - **0.9685**
          - 0.9990
          - 4.0 M
          - 15.45 Mb
        * - *huge*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA

|

Segmentation
------------

A full-glasses segmenter generates masks of people wearing corresponding categories of glasses and their frames, whereas frames-only segmenter generates corresponding masks but only for glasses frames:

.. list-table::
    :header-rows: 1

    * - Model type
      - .. image:: ../../assets/eyeglasses.jpg
      - .. image:: ../../assets/sunglasses.jpg
      - .. image:: ../../assets/no_glasses.jpg
    * - Full/frames eyeglasses segmenter
      - .. image:: ../../assets/eyeglasses_mask.jpg
      - .. image:: ../../assets/no_glasses_mask.jpg
      - .. image:: ../../assets/no_glasses_mask.jpg
    * - Full/frames sunglasses segmenter
      - .. image:: ../../assets/no_glasses_mask.jpg
      - .. image:: ../../assets/sunglasses_mask.jpg
      - .. image:: ../../assets/no_glasses_mask.jpg
    * - Full/frames any glasses segmenter
      - .. image:: ../../assets/eyeglasses_mask.jpg
      - .. image:: ../../assets/sunglasses_mask.jpg
      - .. image:: ../../assets/no_glasses_mask.jpg

There is only one model group for each *full-glasses* and *frames-only* *segmentation* tasks. Each group is trained for both *eyeglasses* and *sunglasses*. Although you can use it as is, it is only one part of the final *full-glasses* or *frames-only* *segmentation* model - the other part is a specific *classifier*, therefore, the accuracy and the model size would be a combination of the generic (base) *segmenter* and a *classifier* of a specific glasses category.

.. collapse:: Full glasses segmentation models (performance & weights)

    .. list-table::
        :header-rows: 1

        * - Model type
          - BCE loss :math:`\downarrow`
          - F1 score :math:`\uparrow`
          - Dice score :math:`\uparrow`
          - Num params :math:`\downarrow`
          - Model size :math:`\downarrow`
        * - *tiny*
          - 0.0580
          - 0.9054
          - 0.9220
          - **926.07 k**
          - **3.54 Mb**
        * - *small*
          - 0.0603
          - 0.8990
          - 0.9131
          - 3.22 M
          - 12.37 Mb
        * - *medium*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *large*
          - **0.0515**
          - **0.9152**
          - **0.9279**
          - 32.95 M
          - 125.89 Mb
        * - *huge*
          - 0.0516
          - 0.9147
          - 0.9272
          - 58.63 M
          - 224.06 Mb

.. collapse:: Glasses frames segmentation models (performance & weights)

    .. list-table::
        :header-rows: 1

        * - Model type
          - BCE loss :math:`\downarrow`
          - F1 score :math:`\uparrow`
          - Dice score :math:`\uparrow`
          - Num params :math:`\downarrow`
          - Model size :math:`\downarrow`
        * - *tiny*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *small*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *medium*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *large*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA
        * - *huge*
          - TBA
          - TBA
          - TBA
          - TBA
          - TBA

|
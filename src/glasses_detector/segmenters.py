import torch
from .bases import BaseSegmenter, _BaseConditionalSegmenter
from .classifiers import EyeglassesClassifier, SunglassesClassifier, GlassesClassifier

class FullGlassesSegmenter(BaseSegmenter):
    """Segmenter to mark the pixels of the glasses.

    A binary segmenter that marks the pixels of full eyeglasses or 
    sunglasses wore by a person in the image. It has 5 different base 
    models ranging from tiny to huge - check the table below for more 
    details:

    .. list-table:: Segmentation backbone model properties
        :header-rows: 1

        * - Backbone name
          - Num parameters ↓
          - Model size ↓
          - BCE loss ↓
          - F1 score ↑
          - ROC-AUC score ↑
        * - TinySegNet (tiny)
          - 926 K
          - 3.704 MB (estimated)
          - 0.0584
          - 0.9031
          - 0.9201
        * - LR-ASPP (small)
          - 3.2 M
          - 12.873 MB (estimated)
          - 0.0595
          - 0.9034
          - 0.9169
        * - FCN (medium)
          - 32.9 M
          - 131.787 MB (estimated)
          - 0.0515
          - 0.9167
          - 0.9292
        * - DeepLab (large)
          - 58.6 M
          - 234.503 MB (estimated)
          - 0.0510
          - 0.9150
          - 0.9275
        * - TBA (huge)
          - TODO
          - TODO
          - TODO
          - TODO
          - TODO

    Note:
        This model always returns binary maps, e.g., a few pixels could 
        be identified as positive, even if the person is not wearing 
        glasses in the image.

    Args:
        base_model (str | torch.nn.Module, optional): The abbreviation 
            of the base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. Finally, it can 
            also be custom torch model, e.g., personally trained on some 
            other data. Defaults to "medium".
        pretrained (bool, optional): Whether to load the pretrained 
            weights for the chosen base model. Check the note inside the 
            documentation of :class:`.BaseModel` to see how the weights 
            are automatically downloaded and loaded. Defaults to False.
    """
    def __init__(
        self, 
        base_model: str | torch.nn.Module = "medium", 
        pretrained: bool = False,
    ):
        super().__init__(base_model, pretrained)

class GlassFramesSegmenter(BaseSegmenter):
    """Segmenter to mark the pixels of the glasses frames.

    A binary segmenter that marks the pixels of eyeglasses or sunglasses 
    frames wore by a person in the image. It has 5 different base models 
    ranging from tiny to huge - check the table below for more details:

    .. list-table:: Segmentation backbone model properties
        :header-rows: 1

        * - Backbone name
          - Num parameters ↓
          - Model size ↓
          - BCE loss ↓
          - F1 score ↑
          - ROC-AUC score ↑
        * - TinySegNet (tiny)
          - 926 K
          - 3.704 MB (estimated)
          - TODO
          - TODO
          - TODO
        * - LR-ASPP (small)
          - 3.2 M
          - 12.873 MB (estimated)
          - TODO
          - TODO
          - TODO
        * - FCN (medium)
          - 32.9 M
          - 131.787 MB (estimated)
          - TODO
          - TODO
          - TODO
        * - DeepLab (large)
          - 58.6 M
          - 234.503 MB (estimated)
          - TODO
          - TODO
          - TODO
        * - TBA (huge)
          - TODO
          - TODO
          - TODO
          - TODO
          - TODO

    Note:
        This model always returns binary maps, e.g., a few pixels could 
        be identified as positive, even if the person is not wearing 
        glasses in the image.
          
    Args:
        base_model (str | torch.nn.Module, optional): The abbreviation 
            of the base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. Finally, it can 
            also be custom torch model, e.g., personally trained on some 
            other data. Defaults to "medium".
        pretrained (bool, optional): Whether to load the pretrained 
            weights for the chosen base model. Check the note inside the 
            documentation of :class:`.BaseModel` to see how the weights 
            are automatically downloaded and loaded. Defaults to False.
    """
    def __init__(
        self, 
        base_model: str | torch.nn.Module = "medium", 
        pretrained: bool = False,
    ):
        super().__init__(base_model, pretrained)

class FullGlassesEyeglassesSegmenter(_BaseConditionalSegmenter):
    """Segmenter to mark the pixels of eyeglasses.

    A binary segmenter that marks the pixels of full eyeglasses (this 
    does not include sunglasses!) wore by a person in the image. It 
    has 5 different base models ranging from tiny to huge for both the 
    classifier that determines if the person is wearing eyeglasses and 
    the full glasses segmenter. Since it is a combination of a 
    classifier and a segmenter, the weights will depend on the 
    architectures chosen for :class:`.EyeglassesClassifier` and for 
    :class:`.FullGlassesSegmenter`. Therefore, the accuracy will be a 
    combination of the chosen architectures.

    Note:
        This model will return empty maps in cases where the person 
        does not wear eyeglasses.

    Args:
        base_model (str | tuple[str | torch.nn.Module, 
            str | torch.nn.Module], optional): The abbreviation of the 
            base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. If provided as a 
            tuple, the first value will be used for the classifier and 
            the second value for segmenter. Note that, in the case of a 
            tuple, it is also possible to provide pure 
            :class:`torch.nn.Module` values. Defaults to "medium".
        pretrained (bool | tuple[bool, bool], optional): Whether to load 
            the pretrained weights for the chosen base model(-s). Check 
            the note inside the documentation of :class:`.BaseModel` to 
            see how the weights are automatically downloaded and loaded. 
            If provided as a tuple, the first value will be used for the 
            classifier and the second value for the segmenter. Defaults 
            to False.
    """
    def __init__(
        self, 
        base_model: str | tuple[str | torch.nn.Module, str | torch.nn.Module], 
        pretrained: bool | tuple[bool, bool] = False,
    ):
        super().__init__(
            EyeglassesClassifier, 
            FullGlassesSegmenter, 
            base_model, 
            pretrained
        )

class FullGlassesSunglassesSegmenter(_BaseConditionalSegmenter):
    """Segmenter to mark the pixels of sunglasses.

    A binary segmenter that marks the pixels of full sunglasses wore by 
    a person in the image. It has 5 different base models ranging from 
    tiny to huge for both the classifier that determines if the person 
    is wearing sunglasses and the full glasses segmenter. Since it is a 
    combination of a classifier and a segmenter, the weights will depend 
    on the architectures chosen for :class:`.SunglassesClassifier` and 
    for :class:`.FullGlassesSegmenter`. Therefore, the accuracy will be 
    a combination of the chosen architectures.

    Note:
        This model will return empty maps in cases where the person 
        does not wear sunglasses.

    Args:
        base_model (str | tuple[str | torch.nn.Module, 
            str | torch.nn.Module], optional): The abbreviation of the 
            base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. If provided as a 
            tuple, the first value will be used for the classifier and 
            the second value for segmenter. Note that, in the case of a 
            tuple, it is also possible to provide pure 
            :class:`torch.nn.Module` values. Defaults to "medium".
        pretrained (bool | tuple[bool, bool], optional): Whether to load 
            the pretrained weights for the chosen base model(-s). Check 
            the note inside the documentation of :class:`.BaseModel` to 
            see how the weights are automatically downloaded and loaded. 
            If provided as a tuple, the first value will be used for the 
            classifier and the second value for the segmenter. Defaults 
            to False.
    """
    def __init__(
        self, 
        base_model: str | tuple[str | torch.nn.Module, str | torch.nn.Module], 
        pretrained: bool | tuple[bool, bool] = False,
    ):
        super().__init__(
            SunglassesClassifier, 
            FullGlassesSegmenter, 
            base_model, 
            pretrained
        )

class FullGlassesGlassesSegmenter(_BaseConditionalSegmenter):
    """Segmenter to mark the pixels of glasses.

    A binary segmenter that marks the pixels of full glasses of any type 
    wore by a person in the image. It has 5 different base models 
    ranging from tiny to huge for both the classifier that determines if 
    the person is wearing any type of glasses and the full glasses 
    segmenter. Since it is a combination of a classifier and a 
    segmenter, the weights will depend on the architectures chosen for 
    :class:`.GlassesClassifier` and for :class:`.FullGlassesSegmenter`. 
    Therefore, the accuracy will be a combination of the chosen 
    architectures.

    Note:
        This model will return empty maps in cases where the person 
        does not wear some type of glasses. Therefore it is different 
        from :class:`FullGlassesSegmenter`.

    Args:
        base_model (str | tuple[str | torch.nn.Module, 
            str | torch.nn.Module], optional): The abbreviation of the 
            base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. If provided as a 
            tuple, the first value will be used for the classifier and 
            the second value for segmenter. Note that, in the case of a 
            tuple, it is also possible to provide pure 
            :class:`torch.nn.Module` values. Defaults to "medium".
        pretrained (bool | tuple[bool, bool], optional): Whether to load 
            the pretrained weights for the chosen base model(-s). Check 
            the note inside the documentation of :class:`.BaseModel` to 
            see how the weights are automatically downloaded and loaded. 
            If provided as a tuple, the first value will be used for the 
            classifier and the second value for the segmenter. Defaults 
            to False.
    """
    def __init__(
        self, 
        base_model: str | tuple[str | torch.nn.Module, str | torch.nn.Module], 
        pretrained: bool | tuple[bool, bool] = False,
    ):
        super().__init__(
            GlassesClassifier, 
            FullGlassesSegmenter, 
            base_model, 
            pretrained
        )

class GlassFramesEyeglassesSegmenter(_BaseConditionalSegmenter):
    """Segmenter to mark the pixels of eyeglasses frames.

    A binary segmenter that marks the pixels of eyeglasses frames (this 
    does not include sunglasses!) wore by a person in the image. It 
    has 5 different base models ranging from tiny to huge for both the 
    classifier that determines if the person is wearing eyeglasses and 
    the glasses frames segmenter. Since it is a combination of a 
    classifier and a segmenter, the weights will depend on the 
    architectures chosen for :class:`.EyeglassesClassifier` and for 
    :class:`.GlassFramesSegmenter`. Therefore, the accuracy will be a 
    combination of the chosen architectures.

    Note:
        This model will return empty maps in cases where the person 
        does not wear eyeglasses.

    Args:
        base_model (str | tuple[str | torch.nn.Module, 
            str | torch.nn.Module], optional): The abbreviation of the 
            base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. If provided as a 
            tuple, the first value will be used for the classifier and 
            the second value for segmenter. Note that, in the case of a 
            tuple, it is also possible to provide pure 
            :class:`torch.nn.Module` values. Defaults to "medium".
        pretrained (bool | tuple[bool, bool], optional): Whether to load 
            the pretrained weights for the chosen base model(-s). Check 
            the note inside the documentation of :class:`.BaseModel` to 
            see how the weights are automatically downloaded and loaded. 
            If provided as a tuple, the first value will be used for the 
            classifier and the second value for the segmenter. Defaults 
            to False.
    """
    def __init__(
        self, 
        base_model: str | tuple[str | torch.nn.Module, str | torch.nn.Module], 
        pretrained: bool | tuple[bool, bool] = False,
    ):
        super().__init__(
            EyeglassesClassifier, 
            GlassFramesSegmenter, 
            base_model, 
            pretrained
        )

class GlassFramesSunglassesSegmenter(_BaseConditionalSegmenter):
    """Segmenter to mark the pixels of sunglasses frames.

    A binary segmenter that marks the pixels of sunglasses frames wore 
    by a person in the image. It has 5 different base models ranging 
    from tiny to huge for both the classifier that determines if the 
    person is wearing sunglasses and the glasses frames segmenter. Since 
    it is a combination of a classifier and a segmenter, the weights 
    will depend on the architectures chosen for 
    :class:`.SunglassesClassifier` and for 
    :class:`.GlassFramesSegmenter`. Therefore, the accuracy will be a 
    combination of the chosen architectures.

    Note:
        This model will return empty maps in cases where the person 
        does not wear sunglasses.

    Args:
        base_model (str | tuple[str | torch.nn.Module, 
            str | torch.nn.Module], optional): The abbreviation of the 
            base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. If provided as a 
            tuple, the first value will be used for the classifier and 
            the second value for segmenter. Note that, in the case of a 
            tuple, it is also possible to provide pure 
            :class:`torch.nn.Module` values. Defaults to "medium".
        pretrained (bool | tuple[bool, bool], optional): Whether to load 
            the pretrained weights for the chosen base model(-s). Check 
            the note inside the documentation of :class:`.BaseModel` to 
            see how the weights are automatically downloaded and loaded. 
            If provided as a tuple, the first value will be used for the 
            classifier and the second value for the segmenter. Defaults 
            to False.
    """
    def __init__(
        self, 
        base_model: str | tuple[str | torch.nn.Module, str | torch.nn.Module], 
        pretrained: bool | tuple[bool, bool] = False,
    ):
        super().__init__(
            SunglassesClassifier, 
            GlassFramesSegmenter, 
            base_model, 
            pretrained
        )

class GlassFramesGlassesSegmenter(_BaseConditionalSegmenter):
    """Segmenter to mark the pixels of glasses frames.

    A binary segmenter that marks the pixels of glasses frames of any 
    type wore by a person in the image. It has 5 different base models 
    ranging from tiny to huge for both the classifier that determines if 
    the person is wearing any type of glasses and the glasses frames 
    segmenter. Since it is a combination of a classifier and a 
    segmenter, the weights will depend on the architectures chosen for 
    :class:`.GlassesClassifier` and for :class:`.GlassFramesSegmenter`. 
    Therefore, the accuracy will be a combination of the chosen 
    architectures.

    Note:
        This model will return empty maps in cases where the person 
        does not wear some type of glasses. Therefore it is different 
        from :class:`GlassFramesSegmenter`.

    Args:
        base_model (str | tuple[str | torch.nn.Module, 
            str | torch.nn.Module], optional): The abbreviation of the 
            base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. If provided as a 
            tuple, the first value will be used for the classifier and 
            the second value for segmenter. Note that, in the case of a 
            tuple, it is also possible to provide pure 
            :class:`torch.nn.Module` values. Defaults to "medium".
        pretrained (bool | tuple[bool, bool], optional): Whether to load 
            the pretrained weights for the chosen base model(-s). Check 
            the note inside the documentation of :class:`.BaseModel` to 
            see how the weights are automatically downloaded and loaded. 
            If provided as a tuple, the first value will be used for the 
            classifier and the second value for the segmenter. Defaults 
            to False.
    """
    def __init__(
        self, 
        base_model: str | tuple[str | torch.nn.Module, str | torch.nn.Module], 
        pretrained: bool | tuple[bool, bool] = False,
    ):
        super().__init__(
            GlassesClassifier, 
            GlassFramesSegmenter, 
            base_model, 
            pretrained
        )

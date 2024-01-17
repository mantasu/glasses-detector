References
==========

The following model architectures were used from `Torchvision <https://pytorch.org/vision/stable/index.html>`_ library:

* **Classifier small** - `ShuffleNet V2 (x0.5) <https://pytorch.org/vision/stable/models/generated/torchvision.models.shufflenet_v2_x0_5.html#torchvision.models.shufflenet_v2_x0_5>`_ based on `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design <https://arxiv.org/abs/1807.11164>`_ paper
* **Classifier medium** - `MobileNet V3 (small) <https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small>`_ based on `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper
* **Classifier large** - `EfficientNet B0 <https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0>`_ based on `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper
* **Segmenter small** - `LRASPP <https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.lraspp_mobilenet_v3_large>`_ based on `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper
* **Segmenter medium** - `FCN (ResNet-50) <https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet50.html#torchvision.models.segmentation.fcn_resnet50>`_ based on `Fully Convolutional Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper
* **Segmenter large** - `DeepLab V3 (ResNet-101) <https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.deeplabv3_resnet101>`_ based on `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`_ paper

:doc:`../modules/glasses_detector.architectures.tiny_binary_classifier`, :doc:`../modules/glasses_detector.architectures.tiny_binary_detector` and :doc:`../modules/glasses_detector.architectures.tiny_binary_segmenter` are the custom models created by me with the aim to have as few parameters as possible while still maintaining a reasonable accuracy.
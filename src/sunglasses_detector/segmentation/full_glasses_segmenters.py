from .base_segmenter import BaseSegmenter, BaseConditionalSegmenter
from ..classification import EyeglassesClassifier, SunglassesClassifier, GlassesClassifier

class FullGlassesSegmenter(BaseSegmenter):
    def __init__(self, base_model: str = "medium", pretrained: bool = False):
        super().__init__(base_model, pretrained)

class FullGlassesEyeglassesSegmenter(BaseConditionalSegmenter):
    def __init__(self, base_model: str | tuple[str, str], pretrained: bool = False):
        super().__init__(EyeglassesClassifier, FullGlassesSegmenter, base_model, pretrained)

class FullGlassesSunglassesSegmenter(BaseConditionalSegmenter):
    def __init__(self, base_model: str | tuple[str, str], pretrained: bool = False):
        super().__init__(SunglassesClassifier, FullGlassesSegmenter, base_model, pretrained)

class FullGlassesGlassesSegmenter(BaseConditionalSegmenter):
    def __init__(self, base_model: str | tuple[str, str], pretrained: bool = False):
        super().__init__(GlassesClassifier, FullGlassesSegmenter, base_model, pretrained)

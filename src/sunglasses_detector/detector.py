import torch
import torch.nn as nn

from segmenter import GlassesSegmenter
from classifier import SunglassesClassifier


class SunglassesDetector(nn.Module):
    def __init__(self, model_type: str | tuple[str, str] = "medium"):
        super().__init__()

        if isinstance(model_type, str):
            # Ensure model type specified for both
            model_type = (model_type, model_type)
        
        # Create classifier and segmenter instances and set to eval state
        # self.classifier = get_sunglasses_classifier(model_type[0]).eval()
        # self.segmenter = get_sunglasses_segmenter(model_type[1]).eval()

        self.classifier = SunglassesClassifier(model_type[0])
        self.segmenter = GlassesSegmenter(model_type[1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict if wearing sunglasses, init masks
        is_sunglasses = self.classifier(x) > 0
        masks = torch.zeros_like(x)

        if not is_sunglasses.any():
            # Empty masks
            return masks
        
        # Get the sunglasses masks for specific images
        sunglasses_masks = self.segmenter(x[is_sunglasses])
        masks[is_sunglasses] = sunglasses_masks

        return masks
    
    def predict(self, image: str | list[str]):
        pass
        
        

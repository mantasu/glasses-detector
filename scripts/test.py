import sys
sys.path.append("src")

import torch
from sunglasses_detector import GlassesSegmenter

segmenter = GlassesSegmenter(base_model="tiny")
segmenter.load_state_dict(torch.load("checkpoints/glasses-segmenter-tiny.pth"))

segmenter.eval()
segmenter.process_dir("data/demo")
import sys
sys.path.append("src")

import torch
from sunglasses_detector.segmentation import FullGlassesSegmenter

model = FullGlassesSegmenter("tiny", pretrained=True)

model.eval()
model.cuda()

model.process("data/backup/demo/example1.jpg")
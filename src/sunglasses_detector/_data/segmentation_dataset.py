from torch.utils.data import Dataset
from mixins import ImageLoaderMixin, DataLoaderMixin

class ImageSegmentationDataset(Dataset, ImageLoaderMixin, DataLoaderMixin):
    def __init__(self):
        super().__init__()
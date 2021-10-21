import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class Coco(Dataset):
    def __init__(self):
        super(Coco, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
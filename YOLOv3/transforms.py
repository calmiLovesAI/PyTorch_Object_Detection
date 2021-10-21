import numpy as np
import torch
import torchvision.transforms.functional as F

from utils import letter_box


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), torch.from_numpy(target)


class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)):
            self.size = size
        else:
            raise TypeError("'size'的类型应该是int，tuple或list其中之一")

    def __call__(self, image, target):
        image, scale, paddings = letter_box(image, self.size)
        target[:, 0:-1] *= scale
        target[:, 0:-1:2] += paddings[2]
        target[:, 1:-1:2] += paddings[0]
        target[:, 0:-1] /= self.size[0]
        return image, target


class TargetPadding:
    def __init__(self, max_num_boxes):
        self.max_num_boxes = max_num_boxes

    def __call__(self, image, target):
        dst = np.full(shape=(self.max_num_boxes, 5), fill_value=-1, dtype=np.float32)
        for i in range(target.shape[0]):
            dst[i] = target[i]
        return image, dst
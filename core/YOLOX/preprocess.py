import cv2
import numpy as np


def resize_with_pad(image, size):
    """

    Args:
        image: numpy.ndarray
        size: Tuple or List, [H, W], 图片resize之后的目标尺寸

    Returns:

    """
    if len(image.shape) == 3:
        padded_img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(size, dtype=np.uint8) * 114

    r = min(size[0] / image.shape[0], size[1] / image.shape[1])
    resized_img = cv2.resize(
        image,
        (int(image.shape[1] * r), int(image.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose((2, 0, 1))   # [H, W, C] -> [C, H, W]
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img




def reverse_resize_with_pad():
    pass
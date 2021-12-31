import cv2
import numpy as np
import torch


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


def reverse_resize_with_pad(h, w, input_size, boxes, xywh=True, coords_normalized=True):
    """

    Args:
        h: 输入网络的图片的原始高度
        w: 输入网络的图片的原始宽度
        input_size: Tuple or List, [H, W], 网络的固定输入图片大小
        boxes: Tensor, shape: (..., 4), 检测框相对于网络的固定输入大小的坐标值
        xywh: Bool, True：boxes是(cx, cy, w, h)格式, False: boxes是(xmin, ymin, xmax, ymax)格式
        coords_normalized: Bool, boxes的坐标值是否已经归一化到[0, 1]

    Returns:

    """
    r = min(input_size[0] / h, input_size[1] / w)
    # 转换为(xmin, ymin, xmax, ymax)格式
    if xywh:
        new_boxes = torch.cat((boxes[..., 0:2] - boxes[..., 2:4] / 2, boxes[..., 0:2] + boxes[..., 2:4] / 2), dim=-1)
    else:
        new_boxes = boxes.clone()
    if coords_normalized:
        new_boxes[..., ::2] *= w
        new_boxes[..., 1::2] *= h
    new_boxes /= r
    return new_boxes

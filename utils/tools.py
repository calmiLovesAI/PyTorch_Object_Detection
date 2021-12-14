import cv2
import torch


def cv2_read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    return image, h, w, c


def letter_box(image, size):
    h, w, _ = image.shape
    H, W = size
    scale = min(H / h, W / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    top = (H - new_h) // 2
    bottom = H - new_h - top
    left = (W - new_w) // 2
    right = W - new_w - left
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                                   value=(128, 128, 128))
    return new_image, scale, [top, bottom, left, right]


def reverse_letter_box(h, w, input_size, boxes):
    """

    :param h: 输入网络的图片的原始高度
    :param w: 输入网络的图片的原始宽度
    :param input_size: 网络的固定输入图片大小
    :param boxes: Tensor, shape: (..., 4(cx, cy, w, h))
    :return: Tensor, shape: (..., 4(xmin, ymin, xmax, ymax))
    """
    # 转换为(xmin, ymin, xmax, ymax)格式
    new_boxes = torch.cat((boxes[..., 0:2] - boxes[..., 2:4] / 2, boxes[..., 0:2] + boxes[..., 2:4] / 2), dim=-1)
    new_boxes *= input_size

    scale = max(h / input_size, w / input_size)
    # 获取padding值
    top = (input_size - h / scale) // 2
    left = (input_size - w / scale) // 2
    # 减去padding值，就是相对于原始图片的原点位置
    new_boxes[..., 0] -= left
    new_boxes[..., 2] -= left
    new_boxes[..., 1] -= top
    new_boxes[..., 3] -= top
    # 缩放到原图尺寸
    new_boxes *= scale
    return new_boxes


def iou_2(anchors, boxes):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors)
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    anchor_max = anchors / 2
    anchor_min = - anchor_max
    box_max = boxes / 2
    box_min = - box_max
    intersect_min = torch.maximum(anchor_min, box_min)
    intersect_max = torch.minimum(anchor_max, box_max)
    intersect_wh = intersect_max - intersect_min
    intersect_wh = torch.clamp(intersect_wh, min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_area = boxes[..., 0] * boxes[..., 1]
    union_area = anchor_area + box_area - intersect_area
    iou = intersect_area / union_area  # shape : [N, 9]
    return iou


class Iou4:
    def __init__(self, box_1, box_2):
        """

        :param box_1: Tensor, shape: (..., 4(cx, cy, w, h))
        :param box_2: Tensor, shape: (..., 4(cx, cy, w, h))
        """
        self.box_1_min, self.box_1_max = Iou4._get_box_min_and_max(box_1)
        self.box_2_min, self.box_2_max = Iou4._get_box_min_and_max(box_2)
        self.box_1_area = box_1[..., 2] * box_1[..., 3]
        self.box_2_area = box_2[..., 2] * box_2[..., 3]

    @staticmethod
    def _get_box_min_and_max(box):
        box_xy = box[..., 0:2]
        box_wh = box[..., 2:4]
        box_min = box_xy - box_wh / 2
        box_max = box_xy + box_wh / 2
        return box_min, box_max

    def calculate_iou(self):
        intersect_min = torch.maximum(self.box_1_min, self.box_2_min)
        intersect_max = torch.minimum(self.box_1_max, self.box_2_max)
        intersect_wh = intersect_max - intersect_min
        intersect_wh = torch.clamp(intersect_wh, min=0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = self.box_1_area + self.box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou


class MeanMetric:
    def __init__(self):
        self.accumulated = 0
        self.count = 0

    def update(self, value):
        self.accumulated += value
        self.count += 1

    def result(self):
        return self.accumulated / self.count

    def reset(self):
        self.__init__()

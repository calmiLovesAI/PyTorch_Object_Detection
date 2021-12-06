from .yolov5_s import YoloV5S
from .yolov5_l import YoloV5L
from .yolov5_m import YoloV5M
from .yolov5_n import YoloV5N
from .yolov5_x import YoloV5X


def get_yolov5_model(name, num_classes, anchors):
    if name == "s":
        return YoloV5S(num_classes, anchors)
    elif name == "l":
        return YoloV5L(num_classes, anchors)
    elif name == "m":
        return YoloV5M(num_classes, anchors)
    elif name == "n":
        return YoloV5N(num_classes, anchors)
    elif name == "x":
        return YoloV5X(num_classes, anchors)
    else:
        raise ValueError("name参数错误")

# models目录下的代码来自：https://github.com/Megvii-BaseDetection/YOLOX
from .darknet import CSPDarknet, Darknet
from .loss import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX

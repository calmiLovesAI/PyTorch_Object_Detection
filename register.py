from scripts import CenterNetTrainer, Yolo3Trainer, Yolo4Trainer, SSDTrainer, YoloXTrainer


class CenterNetCFG:
    name = "centernet"
    cfg_file = "centernet.yaml"

    @staticmethod
    def get_trainer(cfg):
        return CenterNetTrainer(cfg)


class YOLOv3CFG:
    name = "yolov3"
    cfg_file = "yolov3.yaml"

    @staticmethod
    def get_trainer(cfg):
        return Yolo3Trainer(cfg)


class YOLOv4CFG:
    name = "yolov4"
    cfg_file = "yolov4.yaml"

    @staticmethod
    def get_trainer(cfg):
        return Yolo4Trainer(cfg)


class SSDCFG:
    name = "ssd"
    cfg_file = "ssd.yaml"

    @staticmethod
    def get_trainer(cfg):
        return SSDTrainer(cfg)


class YOLOxSCFG:
    name = "yolox_s"
    cfg_file = "yolox_s.py"

    @staticmethod
    def get_trainer(cfg):
        return YoloXTrainer(cfg)


class YOLOxMCFG:
    name = "yolox_m"
    cfg_file = "yolox_m.py"

    @staticmethod
    def get_trainer(cfg):
        return YoloXTrainer(cfg)


class YOLOxLCFG:
    name = "yolox_l"
    cfg_file = "yolox_l.py"

    @staticmethod
    def get_trainer(cfg):
        return YoloXTrainer(cfg)


class YOLOxXCFG:
    name = "yolox_x"
    cfg_file = "yolox_x.py"

    @staticmethod
    def get_trainer(cfg):
        return YoloXTrainer(cfg)
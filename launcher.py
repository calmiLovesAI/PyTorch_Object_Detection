from datetime import datetime

import torch

from load_yaml import load_yamls
from scripts import CenterNetTrainer, Yolo3Trainer, Yolo4Trainer, SSDTrainer


def get_time_format():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


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


if __name__ == '__main__':
    model = SSDCFG

    model_name = model.name
    config = model.cfg_file
    mode = "test"  # "train" or "test"
    model_filename = "SSD_voc_epoch_0.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))
    cfg = load_yamls(model_yaml=config, device=device)

    print("Start {}ing {}.".format(mode, model_name))

    trainer = model.get_trainer(cfg)
    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test(images=cfg["Train"]["test_pictures"], prefix=get_time_format(), model_filename=model_filename, load_model=True)
    else:
        raise ValueError


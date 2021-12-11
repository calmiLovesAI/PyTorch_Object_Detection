from datetime import datetime

import torch

from load_yaml import load_yamls
from scripts import CenterNetTrainer, Yolo3Trainer, Yolo4Trainer


def get_time_format():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == '__main__':
    model_names = ["centernet", "yolov3", "yolov4"]
    cfgs = ["centernet.yaml", "yolov3.yaml", "yolov4.yaml"]
    model_name = model_names[0]
    config = cfgs[0]
    mode = "train"  # "train" or "test"
    model_filename = ""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))
    cfg = load_yamls(model_yaml=config, device=device)

    print("Start {}ing {}.".format(mode, model_name))
    if model_name == "centernet":
        if mode == "train":
            CenterNetTrainer(cfg).train()
        elif mode == "test":
            CenterNetTrainer(cfg).test(images=cfg["Train"]["test_pictures"], prefix=get_time_format(), model_filename=model_filename, load_model=True)
        else:
            raise ValueError
    elif model_name == "yolov3":
        if mode == "train":
            Yolo3Trainer(cfg).train()
        elif mode == "test":
            Yolo3Trainer(cfg).test(images=cfg["Train"]["test_pictures"], prefix=get_time_format(), model_filename=model_filename, load_model=True)
        else:
            raise ValueError
    elif model_name == "yolov4":
        if mode == "train":
            Yolo4Trainer(cfg).train()
        elif mode == "test":
            Yolo4Trainer(cfg).test(images=cfg["Train"]["test_pictures"], prefix=get_time_format(), model_filename=model_filename, load_model=True)
        else:
            raise ValueError
    else:
        raise NotImplementedError

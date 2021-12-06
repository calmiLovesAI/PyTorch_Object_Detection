import torch

from load_yaml import load_yamls
from scripts import CenterNetTrainer, Yolo3Trainer


if __name__ == '__main__':
    model_names = ["centernet", "yolov3"]
    cfgs = ["centernet.yaml", "yolov3.yaml"]
    CONFIG = {
        "model_name": "centernet",
        "cfg": "centernet.yaml"
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))
    cfg = load_yamls(model_yaml=CONFIG["cfg"], device=device)

    if CONFIG["model_name"] == "centernet":
        print("Start training CenterNet.")
        CenterNetTrainer(cfg).train()
    elif CONFIG["model_name"] == "yolov3":
        print("Start training YoloV3.")
        Yolo3Trainer(cfg).train()

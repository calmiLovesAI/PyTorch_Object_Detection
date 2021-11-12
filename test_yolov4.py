import torch
import os
import time

from YOLOv4.inference import test_pipeline
from YOLOv4.load_yaml import load_yaml
from YOLOv4.model import YOLOv4
from datetime import datetime


def get_time_format():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def detect(cfg, model, images, info):
    for image in images:
        start_time = time.time()
        save_dir = "./detect/{}_".format(info) + os.path.basename(image).split(".")[0] + ".jpg"
        test_pipeline(cfg, model, image, save_dir=save_dir)
        print("检测图片{}用时：{:.4f}s".format(image, time.time() - start_time))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    cfg = load_yaml(device)

    model = YOLOv4(cfg["Model"]["num_classes"])
    model.to(device=device)
    model.load_state_dict(torch.load(cfg["Train"]["save_path"] + "YOLOv4.pth", map_location=device))
    model.eval()

    detect(cfg, model, cfg["Train"]["test_pictures"], info=get_time_format())

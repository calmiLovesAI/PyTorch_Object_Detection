import torch
import yaml
import time

from YOLOv3.check import check_cfg
from YOLOv3.inference import test_pipeline
from YOLOv3.model import YoloV3


def detect(cfg, model, images, device):
    for image in images:
        start_time = time.time()
        test_pipeline(cfg, model, image, device)
        print("检测图片{}用时：{:.4f}s".format(image, time.time() - start_time))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    with open(file="experiments/yolov3.yaml") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    check_cfg(cfg)

    model = YoloV3(cfg["Model"]["num_classes"])
    model.to(device=device)
    # model.load_state_dict(torch.load(cfg["Train"]["save_path"] + "YOLOv3.pth"))
    model.eval()

    detect(cfg, model, cfg["Train"]["test_pictures"], device)
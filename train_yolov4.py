import torch

from YOLOv4.dataloader import build_train_loader
from YOLOv4.load_yaml import load_yaml
from YOLOv4.model import YOLOv4

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    cfg = load_yaml(device)

    # 一些训练超参数
    epochs = cfg["Train"]["epochs"]
    batch_size = cfg["Train"]["batch_size"]
    num_classes = cfg["Model"]["num_classes"]
    learning_rate = cfg["Train"]["learning_rate"]

    train_loader = build_train_loader(cfg)

    model = YOLOv4(num_classes)
    model.to(device=device)

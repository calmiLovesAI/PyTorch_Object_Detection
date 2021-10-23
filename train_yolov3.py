import yaml
import torch

from YOLOv3.check import check_cfg
from YOLOv3.dataloader import build_train_loader
from YOLOv3.loss import make_label, YoloLoss
from YOLOv3.model import YoloV3


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    with open(file="experiments/yolov3.yaml") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    check_cfg(cfg)

    # 数据集
    train_loader = build_train_loader(cfg)

    # 模型
    model = YoloV3(cfg["Model"]["num_classes"])
    model.to(device=device)

    # loss
    criterion = YoloLoss(cfg, device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters())

    for epoch in range(cfg["Train"]["epochs"]):
        for i, (img, tar) in enumerate(train_loader):
            print("i = ", i)
            print("img size: {}, tar size: {}".format(img.size(), tar.size()))
            labels = make_label(cfg, tar)
            images = img.to(device=device)
            labels = [x.to(device=device) for x in labels]

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            break
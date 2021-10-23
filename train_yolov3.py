import yaml
import torch
import time

from YOLOv3.check import check_cfg
from YOLOv3.dataloader import build_train_loader
from YOLOv3.loss import make_label, YoloLoss
from YOLOv3.model import YoloV3
from utils import MeanMetric

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    with open(file="experiments/yolov3.yaml") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    check_cfg(cfg)

    # 数据集
    train_loader, train_data = build_train_loader(cfg)
    steps_per_epoch = len(train_data) // cfg["Train"]["batch_size"]

    # 模型
    model = YoloV3(cfg["Model"]["num_classes"])
    model.to(device=device)
    model.train()

    # loss
    criterion = YoloLoss(cfg, device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters())

    loss_mean = MeanMetric()
    loc_loss_mean = MeanMetric()
    conf_loss_mean = MeanMetric()
    prob_loss_mean = MeanMetric()

    for epoch in range(cfg["Train"]["epochs"]):
        for i, (img, tar) in enumerate(train_loader):
            start_time = time.time()

            labels = make_label(cfg, tar)
            images = img.to(device=device)
            labels = [x.to(device=device) for x in labels]

            optimizer.zero_grad()
            preds = model(images)
            loss, loc_loss, conf_loss, prob_loss = criterion(preds, labels)
            loss_mean.update(loss)
            loc_loss_mean.update(loc_loss)
            conf_loss_mean.update(conf_loss)
            prob_loss_mean.update(prob_loss)
            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, step: {}/{}, speed: {:.3f}s/step, total_loss: {}, "
                  "loc_loss: {}, conf_loss: {}, prob_loss: {}".format(epoch,
                                                                      cfg["Train"]["epochs"],
                                                                      i,
                                                                      steps_per_epoch,
                                                                      time.time() - start_time,
                                                                      loss_mean.result(),
                                                                      loc_loss_mean.result(),
                                                                      conf_loss_mean.result(),
                                                                      prob_loss_mean.result(),
                                                                      ))

        loss_mean.reset()
        loc_loss_mean.reset()
        conf_loss_mean.reset()
        prob_loss_mean.reset()


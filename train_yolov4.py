import time

import torch

from YOLOv4.dataloader import build_train_loader
from YOLOv4.load_yaml import load_yaml
from YOLOv4.loss import YoloLoss, make_label
from YOLOv4.model import YOLOv4
from utils.tools import MeanMetric

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

    criterion = YoloLoss(cfg)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # 学习率调整
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    loss_mean = MeanMetric()
    loc_loss_mean = MeanMetric()
    conf_loss_mean = MeanMetric()
    prob_loss_mean = MeanMetric()

    for epoch in range(0, epochs):
        model.train()
        for i, (img, tar) in enumerate(train_loader):
            start_time = time.time()

            images = img.to(device=device)
            tar = tar.to(device=device)
            labels = make_label(cfg, tar)

            optimizer.zero_grad()
            preds = model(images)
            loss, loc_loss, conf_loss, prob_loss = criterion(preds, labels)
            loss_mean.update(loss.item())
            loc_loss_mean.update(loc_loss.item())
            conf_loss_mean.update(conf_loss.item())
            prob_loss_mean.update(prob_loss.item())
            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, step: {}/{}, speed: {:.3f}s/step, total_loss: {}, "
                  "loc_loss: {}, conf_loss: {}, prob_loss: {}".format(epoch,
                                                                      epochs,
                                                                      i,
                                                                      len(train_loader),
                                                                      time.time() - start_time,
                                                                      loss_mean.result(),
                                                                      loc_loss_mean.result(),
                                                                      conf_loss_mean.result(),
                                                                      prob_loss_mean.result(),
                                                                      ))

        scheduler.step(loss_mean.result())

        loss_mean.reset()
        loc_loss_mean.reset()
        conf_loss_mean.reset()
        prob_loss_mean.reset()


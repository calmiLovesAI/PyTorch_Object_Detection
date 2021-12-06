import time
from pathlib import Path

import torch

from core.YOLOv4.dataloader import TrainLoader
from load_yaml import load_yamls
from core.YOLOv4.loss import YoloLoss, make_label
from core.YOLOv4 import YOLOv4
from test_yolov4 import detect
from utils.tools import MeanMetric

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    cfg = load_yamls(model_yaml="yolov4.yaml", device=device)

    # 一些训练超参数
    epochs = cfg["Train"]["epochs"]
    batch_size = cfg["Train"]["batch_size"]
    num_classes = cfg["Model"]["num_classes"]
    learning_rate = cfg["Train"]["learning_rate"]
    save_frequency = cfg["Train"]["save_frequency"]  # 模型保存频率
    save_path = cfg["Train"]["save_path"]  # 模型保存路径
    test_pictures = cfg["Train"]["test_pictures"]  # 测试图片路径列表
    load_weights = cfg["Train"]["load_weights"]  # 训练之前是否加载权重
    test_during_training = cfg["Train"]["test_during_training"]  # 是否在每一轮epoch结束后开启图片测试
    resume_training_from_epoch = cfg["Train"]["resume_training_from_epoch"]
    tensorboard_on = cfg["Train"]["tensorboard_on"]  # 是否开启tensorboard

    train_loader = TrainLoader(cfg).__call__()

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

    start_epoch = -1
    if load_weights:
        saved_model = Path(save_path).joinpath("YOLOv4_epoch_{}.pth".format(resume_training_from_epoch))
        model.load_state_dict(torch.load(saved_model, map_location=device))
        start_epoch = resume_training_from_epoch

    if tensorboard_on:
        writer = SummaryWriter()   # tensorboard --logdir=runs
        writer.add_graph(model, torch.randn(batch_size, 3, cfg["Train"]["input_size"], cfg["Train"]["input_size"],
                                            dtype=torch.float32, device=device))

    for epoch in range(start_epoch+1, epochs):
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
            if tensorboard_on:
                writer.add_scalar(tag="Total Loss", scalar_value=loss_mean.result(),
                                  global_step=epoch * len(train_loader) + i)
                writer.add_scalar(tag="Loc Loss", scalar_value=loc_loss_mean.result(),
                                  global_step=epoch * len(train_loader) + i)
                writer.add_scalar(tag="Conf Loss", scalar_value=conf_loss_mean.result(),
                                  global_step=epoch * len(train_loader) + i)
                writer.add_scalar(tag="Prob Loss", scalar_value=prob_loss_mean.result(),
                                  global_step=epoch * len(train_loader) + i)

        scheduler.step(loss_mean.result())

        loss_mean.reset()
        loc_loss_mean.reset()
        conf_loss_mean.reset()
        prob_loss_mean.reset()

        if epoch % save_frequency == 0:
            torch.save(model.state_dict(), save_path + "YOLOv4_epoch_{}.pth".format(epoch))

        if test_during_training:
            model.eval()
            detect(cfg, model, test_pictures, info="epoch-{}".format(epoch))

    if tensorboard_on:
        writer.close()
    torch.save(model.state_dict(), save_path + "YOLOv4_weights.pth")
    torch.save(model, save_path + "YOLOv4_entire_model.pth")


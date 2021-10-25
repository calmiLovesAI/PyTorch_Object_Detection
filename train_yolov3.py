import yaml
import torch
import time

from pathlib import Path
from YOLOv3.check import check_cfg
from YOLOv3.dataloader import build_train_loader
from YOLOv3.loss import make_label, YoloLoss
from YOLOv3.model import YoloV3
from utils import MeanMetric
from test_yolov3 import detect

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    with open(file="experiments/yolov3.yaml") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    check_cfg(cfg)

    # 一些训练超参数
    epochs = cfg["Train"]["epochs"]
    batch_size = cfg["Train"]["batch_size"]
    num_classes = cfg["Model"]["num_classes"]
    learning_rate = cfg["Train"]["learning_rate"]
    save_frequency = cfg["Train"]["save_frequency"]   # 模型保存频率
    save_path = cfg["Train"]["save_path"]   # 模型保存路径
    test_pictures = cfg["Train"]["test_pictures"]   # 测试图片路径列表
    load_weights = cfg["Train"]["load_weights"]    # 训练之前是否加载权重
    test_during_training = cfg["Train"]["test_during_training"]    # 是否在每一轮epoch结束后开启图片测试
    resume_training_from_epoch = cfg["Train"]["resume_training_from_epoch"]

    # tensorboard --logdir=runs
    writer = SummaryWriter()

    # 数据集
    train_loader = build_train_loader(cfg)

    # 模型
    model = YoloV3(num_classes)
    model.to(device=device)

    # loss
    criterion = YoloLoss(cfg, device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    loss_mean = MeanMetric()
    loc_loss_mean = MeanMetric()
    conf_loss_mean = MeanMetric()
    prob_loss_mean = MeanMetric()

    start_epoch = -1

    if load_weights:
        saved_model = Path(save_path).joinpath("YOLOv3_epoch_{}.pth".format(resume_training_from_epoch))
        model.load_state_dict(torch.load(saved_model, map_location=device))
        start_epoch = resume_training_from_epoch

    for epoch in range(start_epoch+1, epochs):
        model.train()
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
                                                                      epochs,
                                                                      i,
                                                                      len(train_loader),
                                                                      time.time() - start_time,
                                                                      loss_mean.result(),
                                                                      loc_loss_mean.result(),
                                                                      conf_loss_mean.result(),
                                                                      prob_loss_mean.result(),
                                                                      ))
            writer.add_graph(model, images)
            writer.add_scalar(tag="Total Loss", scalar_value=loss_mean.result(),
                              global_step=epoch * len(train_loader) + i)
            writer.add_scalar(tag="Loc Loss", scalar_value=loc_loss_mean.result(),
                              global_step=epoch * len(train_loader) + i)
            writer.add_scalar(tag="Conf Loss", scalar_value=conf_loss_mean.result(),
                              global_step=epoch * len(train_loader) + i)
            writer.add_scalar(tag="Prob Loss", scalar_value=prob_loss_mean.result(),
                              global_step=epoch * len(train_loader) + i)

        loss_mean.reset()
        loc_loss_mean.reset()
        conf_loss_mean.reset()
        prob_loss_mean.reset()

        if epoch % save_frequency == 0:
            save_path = save_path + "YOLOv3_epoch_{}.pth".format(epoch)
            torch.save(model.state_dict(), save_path)

        if test_during_training:
            model.eval()
            detect(cfg, model, test_pictures, device, info="epoch-{}".format(epoch))

    writer.close()
    torch.save(model.state_dict(), save_path + "YOLOv3.pth")


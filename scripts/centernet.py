import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from CenterNet.dataloader import TrainLoader
from CenterNet.loss import CombinedLoss
from CenterNet.model import CenterNet
from CenterNet.target_generator import TargetGenerator
from utils.tools import MeanMetric
from .template import ITrainer


class CenterNetTrainer(ITrainer):
    def __init__(self, cfg):
        # 一些训练超参数
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.cfg = cfg
        self.device = cfg["device"]
        self.epochs = cfg["Train"]["epochs"]
        self.batch_size = cfg["Train"]["batch_size"]
        self.input_size = cfg["Train"]["input_size"]
        self.num_classes = cfg["Model"]["num_classes"]
        self.learning_rate = cfg["Train"]["learning_rate"]
        self.save_frequency = cfg["Train"]["save_frequency"]  # 模型保存频率
        self.save_path = cfg["Train"]["save_path"]  # 模型保存路径
        self.test_pictures = cfg["Train"]["test_pictures"]  # 测试图片路径列表
        self.load_weights = cfg["Train"]["load_weights"]  # 训练之前是否加载权重
        self.test_during_training = cfg["Train"]["test_during_training"]  # 是否在每一轮epoch结束后开启图片测试
        self.resume_training_from_epoch = cfg["Train"]["resume_training_from_epoch"]
        self.tensorboard_on = cfg["Train"]["tensorboard_on"]  # 是否开启tensorboard

        # 训练数据集
        self.train_dataloader = None

    def set_model(self):
        self.model = CenterNet(self.cfg)
        self.model.to(device=self.device)

    def set_train_dataloader(self, *args, **kwargs):
        self.train_dataloader = TrainLoader(self.cfg).__call__()

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def set_lr_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2)

    def load(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def save(self, epoch, save_entire_model=False):
        torch.save(self.model.state_dict(), self.save_path + "centernet_epoch_{}.pth".format(epoch))
        if save_entire_model:
            torch.save(self.model, self.save_path + "centernet_entire_model.pth")

    def train(self, *args, **kwargs):
        self.set_model()
        self.set_train_dataloader()
        self.set_optimizer()
        self.set_lr_scheduler()
        # 损失函数
        criterion = CombinedLoss(self.cfg)
        # metrics
        loss_mean = MeanMetric()
        start_epoch = 0
        if self.load_weights:
            # 加载权重参数
            self.load(weights_path=Path(self.save_path).joinpath(
                "centernet_epoch_{}.pth".format(self.resume_training_from_epoch)))
            start_epoch = self.resume_training_from_epoch
        if self.tensorboard_on:
            writer = SummaryWriter()  # 在控制台使用命令tensorboard --logdir=runs进入tensorboard面板
            writer.add_graph(self.model, torch.randn(self.batch_size, 3, self.input_size, self.input_size,
                                                     dtype=torch.float32, device=self.device))
        for epoch in range(start_epoch + 1, self.epochs):
            self.model.train()
            for i, (images, labels) in enumerate(self.train_dataloader):
                start_time = time.time()

                images = images.to(device=self.device)
                labels = labels.to(device=self.device)
                target = TargetGenerator(self.cfg, labels).__call__()

                self.optimizer.zero_grad()
                preds = self.model(images)
                target.insert(0, preds)
                loss = criterion(*target)
                loss_mean.update(loss.item())
                loss.backward()
                self.optimizer.step()

                print("Epoch: {}/{}, step: {}/{}, speed: {:.3f}s/step, loss: {}".format(epoch,
                                                                          self.epochs,
                                                                          i,
                                                                          len(self.train_dataloader),
                                                                          time.time() - start_time,
                                                                          loss_mean.result(),
                                                                          ))
                if self.tensorboard_on:
                    writer.add_scalar(tag="Loss", scalar_value=loss_mean.result(),
                                      global_step=epoch * len(self.train_dataloader) + i)
            self.scheduler.step(loss_mean.result())
            loss_mean.reset()

            if epoch % self.save_frequency == 0:
                self.save(epoch=epoch)

        self.save(epoch=self.epochs, save_entire_model=True)

    def test(self, *args, **kwargs):
        pass

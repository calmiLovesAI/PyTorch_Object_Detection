import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from core.YOLOv3.dataloader import TrainLoader
from core.YOLOv3.inference import Inference
from core.YOLOv3.loss import YoloLoss, make_label
from core.YOLOv3.model import YoloV3
from dataset import find_class_name
from draw import Draw
from utils.tools import MeanMetric, letter_box
from .template import ITrainer


class Yolo3Trainer(ITrainer):
    def __init__(self, cfg):
        # 一些训练超参数
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.cfg = cfg
        self.device = cfg["device"]
        self.dataset_name = cfg["Train"]["dataset_name"]
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

    def _set_model(self):
        self.model = YoloV3(self.num_classes)
        self.model.to(device=self.device)

    def _set_train_dataloader(self, *args, **kwargs):
        self.train_dataloader = TrainLoader(self.cfg).__call__()

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def _set_lr_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2)

    def _load(self, weights_path):
        if self.model is None:
            self._set_model()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def _save(self, epoch, save_entire_model=False):
        torch.save(self.model.state_dict(), self.save_path + "YOLOv3_epoch_{}.pth".format(epoch))
        if save_entire_model:
            torch.save(self.model, self.save_path + "YOLOv3_entire_model.pth")

    def train(self, *args, **kwargs):
        self._set_model()
        self._set_train_dataloader()
        self._set_optimizer()
        self._set_lr_scheduler()
        # 损失函数
        criterion = YoloLoss(self.cfg, self.device)
        # metrics
        loss_mean = MeanMetric()
        loc_loss_mean = MeanMetric()
        conf_loss_mean = MeanMetric()
        prob_loss_mean = MeanMetric()

        start_epoch = -1
        if self.load_weights:
            # 加载权重参数
            self._load(weights_path=Path(self.save_path).joinpath(
                "YOLOv3_epoch_{}.pth".format(self.resume_training_from_epoch)))
            start_epoch = self.resume_training_from_epoch

        if self.tensorboard_on:
            writer = SummaryWriter()  # 在控制台使用命令 tensorboard --logdir=runs 进入tensorboard面板
            writer.add_graph(self.model, torch.randn(self.batch_size, 3, self.input_size, self.input_size,
                                                     dtype=torch.float32, device=self.device))
        for epoch in range(start_epoch + 1, self.epochs):
            self.model.train()
            for i, (images, labels) in enumerate(self.train_dataloader):
                start_time = time.time()

                images = images.to(device=self.device)
                targets = make_label(self.cfg, labels)
                targets = [x.to(device=self.device) for x in targets]

                self.optimizer.zero_grad()
                preds = self.model(images)
                loss, loc_loss, conf_loss, prob_loss = criterion(preds, targets)
                loss_mean.update(loss.item())
                loc_loss_mean.update(loc_loss.item())
                conf_loss_mean.update(conf_loss.item())
                prob_loss_mean.update(prob_loss.item())
                loss.backward()
                self.optimizer.step()

                print("Epoch: {}/{}, step: {}/{}, speed: {:.3f}s/step, total_loss: {}, "
                      "loc_loss: {}, conf_loss: {}, prob_loss: {}".format(epoch,
                                                                          self.epochs,
                                                                          i,
                                                                          len(self.train_dataloader),
                                                                          time.time() - start_time,
                                                                          loss_mean.result(),
                                                                          loc_loss_mean.result(),
                                                                          conf_loss_mean.result(),
                                                                          prob_loss_mean.result(),
                                                                          ))
                if self.tensorboard_on:
                    writer.add_scalar(tag="Total Loss", scalar_value=loss_mean.result(),
                                      global_step=epoch * len(self.train_dataloader) + i)
                    writer.add_scalar(tag="Loc Loss", scalar_value=loc_loss_mean.result(),
                                      global_step=epoch * len(self.train_dataloader) + i)
                    writer.add_scalar(tag="Conf Loss", scalar_value=conf_loss_mean.result(),
                                      global_step=epoch * len(self.train_dataloader) + i)
                    writer.add_scalar(tag="Prob Loss", scalar_value=prob_loss_mean.result(),
                                      global_step=epoch * len(self.train_dataloader) + i)

            self.scheduler.step(loss_mean.result())
            loss_mean.reset()
            loc_loss_mean.reset()
            conf_loss_mean.reset()
            prob_loss_mean.reset()

            if epoch % self.save_frequency == 0:
                self._save(epoch=epoch)

            if self.test_during_training:
                self.test(images=self.test_pictures, prefix="epoch-{}".format(epoch), model_filename="")

        if self.tensorboard_on:
            writer.close()

        self._save(epoch=self.epochs, save_entire_model=True)

    def test(self, images, prefix, model_filename, load_model=False, *args, **kwargs):
        if load_model:
            self._load(weights_path=Path(self.save_path).joinpath(model_filename))
        self.model.eval()
        for image in images:
            start_time = time.time()
            save_dir = "./detect/{}_".format(prefix) + os.path.basename(image).split(".")[0] + ".jpg"
            self._test_pipeline(image, save_dir=save_dir)
            print("检测图片{}用时：{:.4f}s".format(image, time.time() - start_time))

    def _test_pipeline(self, image_path, save_dir=None, print_on=True, save_result=True, *args, **kwargs):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        image, _, _ = letter_box(image, (self.input_size, self.input_size))
        image = to_tensor(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(device=self.device)

        with torch.no_grad():
            outputs = self.model(image)
            boxes, scores, classes = Inference(cfg=self.cfg, outputs=outputs, input_image_shape=(h, w),
                                               device=self.device).get_results()
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        scores = np.squeeze(scores, axis=0)
        classes = classes.cpu().numpy().tolist()
        classes = [find_class_name(self.dataset_name, c, keep_index=True) for c in classes]
        if print_on:
            print("检测出{}个边界框，分别是：".format(boxes.shape[0]))
            print("boxes: ", boxes)
            print("scores: ", scores)
            print("classes: ", classes)

        painter = Draw()
        image_with_boxes = painter.draw_boxes_on_image(image_path, boxes, scores, classes)

        if save_result:
            # 保存检测结果
            cv2.imwrite(save_dir, image_with_boxes)
        else:
            return image_with_boxes

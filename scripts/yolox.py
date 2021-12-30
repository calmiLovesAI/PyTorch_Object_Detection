import os
import time
from pathlib import Path

import cv2
import torch
from torchvision.transforms.functional import to_tensor

from core.YOLOX.inference import postprocess, get_specific_detection_results
from dataset import find_class_name
from draw import Draw
from scripts.template import ITrainer
from experiments.yolox_base import BaseExp
from core.YOLOX.models import YOLOX, YOLOPAFPN, YOLOXHead
from utils.tools import cv2_read_image, direct_image_resize


class YoloXTrainer(ITrainer):
    def __init__(self, cfg: BaseExp):
        # 一些训练超参数
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.cfg = cfg
        self.device = cfg.device
        self.start_epoch = cfg.start_epoch
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.input_size = cfg.input_size
        self.model_name = cfg.model_name
        self.learning_rate = cfg.learning_rate
        self.dataset_name = cfg.dataset_name
        self.save_frequency = cfg.save_frequency  # 模型保存频率
        self.save_path = cfg.save_path  # 模型保存路径
        self.test_pictures = cfg.test_pictures  # 测试图片路径列表
        self.load_weights = cfg.load_weights  # 训练之前是否加载权重
        self.test_during_training = cfg.test_during_training  # 是否在每一轮epoch结束后开启图片测试
        self.pretrained_weights = cfg.pretrained_weights  # 预训练的模型路径
        self.tensorboard_on = cfg.tensorboard_on  # 是否开启tensorboard

        # 训练数据集
        self.train_dataloader = None

        # 网络结构
        self.depth = cfg.depth
        self.width = cfg.width
        self.num_classes = cfg.num_classes
        self.act = cfg.act

        self.nms_threshold = cfg.nms_threshold
        self.conf_threshold = cfg.confidence_threshold

    def _set_model(self, *args, **kwargs):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def _set_lr_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2)

    def _load(self, weights_path):
        if self.model is None:
            self._set_model()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def _save(self, epoch, save_entire_model=False):
        torch.save(self.model.state_dict(),
                   self.save_path + "{}_{}_epoch_{}.pth".format(self.model_name, self.dataset_name, epoch))
        if save_entire_model:
            torch.save(self.model, self.save_path + "{}_{}_entire_model.pth".format(self.model_name, self.dataset_name))

    def test(self, images, prefix, model_filename, load_model=False, *args, **kwargs):
        if load_model:
            self._load(weights_path=Path(self.save_path).joinpath(model_filename))
        self.model.eval()
        for image in images:
            t0 = time.time()
            save_dir = "./detect/{}_".format(prefix) + os.path.basename(image).split(".")[0] + ".jpg"
            self._test_pipeline(image, save_dir=save_dir)
            print("检测图片{}用时：{:.4f}s".format(image, time.time() - t0))

    def _test_pipeline(self, image_path, save_dir=None, print_on=True, save_result=True, *args, **kwargs):
        image, h, w, c = cv2_read_image(image_path)
        image, _, _ = direct_image_resize(image, (self.input_size, self.input_size))
        image = to_tensor(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(device=self.device)

        with torch.no_grad():
            outputs = self.model(image)
            detections = postprocess(outputs, self.num_classes, self.conf_threshold, self.nms_threshold, class_agnostic=True)
            boxes, scores, classes = get_specific_detection_results(detections, h, w, self.input_size)
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy().tolist()
        classes = [find_class_name(self.cfg, c, keep_index=True) for c in classes]
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


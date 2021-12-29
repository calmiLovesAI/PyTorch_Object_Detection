from scripts.template import ITrainer
from experiments.yolox_base import BaseExp


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


    def _set_model(self, *args, **kwargs):
        pass
class BaseExp:
    def __init__(self, device):
        self.device = device
        self.start_epoch = 0
        self.epochs = 100
        self.batch_size = 8
        self.learning_rate = 0.001
        self.input_size = 640
        # "voc" or "coco"
        self.dataset_name = "coco"
        self.max_num_boxes = 30
        self.save_path = "./saved_model/"
        self.save_frequency = 1
        self.test_during_training = True
        self.load_weights = False
        self.pretrained_weights = ""
        self.tensorboard_on = True
        self.test_pictures = [""]

        # model
        self.model_name = "yolox"
        self.num_classes = self._get_num_classes()
        self.depth = 1.00
        self.width = 1.00
        self.act = "silu"

        # test
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.01

    def _get_num_classes(self):
        if self.dataset_name == "coco":
            return 80
        elif self.dataset_name == "voc":
            return 20
        else:
            raise NotImplementedError

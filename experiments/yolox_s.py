from .yolox_base import BaseExp


class Exp(BaseExp):
    def __init__(self, device):
        super(Exp, self).__init__(device)
        self.depth = 0.33
        self.width = 0.50
        self.model_name = "yolox_s"

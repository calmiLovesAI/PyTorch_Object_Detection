from .yolox_base import BaseExp


class Exp(BaseExp):
    def __init__(self, device):
        super(Exp, self).__init__(device)
        self.depth = 1.0
        self.width = 1.0
        self.model_name = "yolox_l"

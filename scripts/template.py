from abc import ABCMeta, abstractmethod


class ITrainer(metaclass=ABCMeta):
    @abstractmethod
    def set_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_lr_scheduler(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass
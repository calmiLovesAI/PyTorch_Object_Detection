from abc import ABCMeta, abstractmethod


class ITrainer(metaclass=ABCMeta):
    @abstractmethod
    def _set_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_train_dataloader(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_lr_scheduler(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def _save(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def _test_pipeline(self, *args, **kwargs):
        """
        已废弃
        Args:
            *args:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def forward_pipeline(self, *args, **kwargs):
        pass

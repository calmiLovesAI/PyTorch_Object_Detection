from dataset.public_dataloader import PublicTrainLoader


class TrainLoader(PublicTrainLoader):
    def __init__(self, cfg):
        super(TrainLoader, self).__init__(cfg)
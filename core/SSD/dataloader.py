from dataset.public_dataloader import PublicTrainLoader


class UpdateClassIndices:
    def __call__(self, image, target):
        target[..., -1] += 1
        return image, target


class TrainLoader(PublicTrainLoader):
    def __init__(self, cfg):
        super(TrainLoader, self).__init__(cfg)
        self.transforms.append(UpdateClassIndices())

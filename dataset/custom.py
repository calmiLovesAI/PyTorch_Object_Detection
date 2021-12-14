from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, cfg, transform=None):
        super(CustomDataset, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
import YOLOv3.transforms as T
from voc import Voc
from torch.utils.data import DataLoader


def build_train_loader(cfg):
    if cfg["Train"]["dataset_name"] == "voc":
        dataset = Voc(cfg["VOC"], T.Compose(transforms=[
            T.Resize(size=cfg["Train"]["input_size"]),
            T.TargetPadding(max_num_boxes=cfg["Train"]["max_num_boxes"]),
            T.ToTensor()
        ]))
    return DataLoader(dataset=dataset, batch_size=cfg["Train"]["batch_size"], shuffle=True)
import YOLOv3.transforms as T
from voc import Voc
from coco import Coco
from torch.utils.data import DataLoader


def build_train_loader(cfg):
    if cfg["Train"]["dataset_name"] == "voc":
        dataset = Voc(cfg["VOC"], T.Compose(transforms=[
            T.Resize(size=cfg["Train"]["input_size"]),
            T.TargetPadding(max_num_boxes=cfg["Train"]["max_num_boxes"]),
            T.ToTensor()
        ]))

    elif cfg["Train"]["dataset_name"] == "coco":
        dataset = Coco(cfg["COCO"], T.Compose(transforms=[
            T.Resize(size=cfg["Train"]["input_size"]),
            T.TargetPadding(max_num_boxes=cfg["Train"]["max_num_boxes"]),
            T.ToTensor()
        ]))
    else:
        raise ValueError("参数cfg->Train->dataset_name错误")
    return DataLoader(dataset=dataset, batch_size=cfg["Train"]["batch_size"], shuffle=True)
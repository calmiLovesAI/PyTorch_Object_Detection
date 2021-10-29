import yaml


def check_cfg(cfg):
    dataset = cfg["Train"]["dataset_name"]
    if dataset == "voc":
        assert cfg["VOC"]["num_classes"] == cfg["Model"]["num_classes"]
        assert cfg["VOC"]["num_classes"] == len(cfg["VOC"]["classes"])
    elif dataset == "coco":
        assert cfg["COCO"]["num_classes"] == cfg["Model"]["num_classes"]
        assert cfg["COCO"]["num_classes"] == len(cfg["COCO"]["classes"])
    else:
        raise ValueError("参数cfg->Train->dataset_name错误")


def load_yaml():
    with open(file="experiments/yolov3.yaml") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(file="experiments/VOC.yaml") as f:
        voc_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(file="experiments/COCO.yaml") as f:
        coco_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["VOC"] = voc_cfg
    cfg["COCO"] = coco_cfg
    check_cfg(cfg)

    return cfg
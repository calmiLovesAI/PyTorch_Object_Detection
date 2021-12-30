import yaml
from pathlib import Path
from experiments import YoloxSCfg


def load_yaml(file):
    with open(file=file) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_yamls(model_yaml, device=None, model_name=None):
    if model_name == "yolox_s":
        return YoloxSCfg(device)
    cfg = load_yaml(Path("./experiments").joinpath(model_yaml))
    if device:
        cfg["device"] = device
    cfg["VOC"] = load_yaml(Path("./experiments").joinpath("VOC.yaml"))
    cfg["COCO"] = load_yaml(Path("./experiments").joinpath("COCO.yaml"))
    cfg["Custom"] = load_yaml(Path("./experiments").joinpath("Custom.yaml"))
    check_cfg(cfg)
    return cfg


def check_cfg(cfg):
    dataset = cfg["Train"]["dataset_name"]
    if dataset == "voc":
        assert cfg["VOC"]["num_classes"] == cfg["Model"]["num_classes"]
        assert cfg["VOC"]["num_classes"] == len(cfg["VOC"]["classes"])
    elif dataset == "coco":
        assert cfg["COCO"]["num_classes"] == cfg["Model"]["num_classes"]
        assert cfg["COCO"]["num_classes"] == len(cfg["COCO"]["classes"])
    elif dataset == "custom":
        assert cfg["Custom"]["num_classes"] == cfg["Model"]["num_classes"]
        assert cfg["Custom"]["num_classes"] == len(cfg["Custom"]["classes"])
    else:
        raise ValueError("参数cfg->Train->dataset_name错误")

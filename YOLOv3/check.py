

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
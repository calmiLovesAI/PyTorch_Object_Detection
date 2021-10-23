

def check_cfg(cfg):
    dataset = cfg["Train"]["dataset_name"]
    if dataset == "voc":
        assert cfg["VOC"]["num_classes"] == cfg["Model"]["num_classes"]
    elif dataset == "COCO":
        assert cfg["COCO"]["num_classes"] == cfg["Model"]["num_classes"]
    else:
        raise ValueError("参数cfg->Train->dataset_name错误")
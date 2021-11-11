import torch


def get_anchor(cfg):
    if cfg["Train"]["dataset_name"] == "voc":
        anchors = cfg["Anchor"]["voc_anchors"]
    if cfg["Train"]["dataset_name"] == "coco":
        anchors = cfg["Anchor"]["coco_anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=cfg["device"])
    anchors = torch.reshape(anchors, shape=(-1, 3, 2))
    # 归一化
    anchors /= cfg["Train"]["input_size"]
    return anchors
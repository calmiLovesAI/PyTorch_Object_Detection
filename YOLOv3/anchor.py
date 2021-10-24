import torch


def get_anchor(cfg, i, device):
    anchors = cfg["Train"]["anchor"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = torch.reshape(anchors, shape=(-1, 2))
    # 归一化
    anchors /= cfg["Train"]["input_size"]
    return anchors[3 * i: 3 * (i + 1), :]

import torch


def generate_grid_index(length, device):
    X = torch.arange(start=0, end=length, step=1, dtype=torch.float32, device=device)
    Y = torch.arange(start=0, end=length, step=1, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(X, Y)
    X = torch.reshape(X, shape=(-1, 1))
    Y = torch.reshape(Y, shape=(-1, 1))
    return torch.cat((X, Y), dim=-1)


def predict_bounding_bbox(cfg, feature_map, anchors, idx, device, is_training=False):
    num_classes = cfg["Model"]["num_classes"]
    N, C, H, W = feature_map.size()
    feature_map = torch.reshape(feature_map, shape=(N, H, W, -1))
    area = H * W
    pred = torch.reshape(feature_map, shape=(N, area * 3, -1))
    tx_ty, tw_th, confidence, class_prob = torch.split(pred, split_size_or_sections=[2, 2, 1, num_classes], dim=-1)
    confidence = torch.sigmoid(confidence)
    class_prob = torch.sigmoid(class_prob)

    center_index = generate_grid_index(length=H, device=device)
    center_index = torch.tile(center_index, dims=[1, 3])
    center_index = torch.reshape(center_index, shape=(1, -1, 2))

    center_coord = center_index + torch.sigmoid(tx_ty)
    box_xy = center_coord / H
    anchors = anchors[idx * 3:(idx+1) * 3, :]
    anchors /= cfg["Train"]["input_size"]
    anchors = torch.tile(anchors, dims=[area, 1])
    bw_bh = anchors * torch.exp(tw_th)
    box_wh = bw_bh

    # reshape
    center_index = torch.reshape(center_index, shape=(-1, H, W, 3, 2))
    box_xy = torch.reshape(box_xy, shape=(-1, H, W, 3, 2))
    box_wh = torch.reshape(box_wh, shape=(-1, H, W, 3, 2))
    feature_map = torch.reshape(feature_map, shape=(-1, H, W, 3, num_classes + 5))

    if is_training:
        return box_xy, box_wh, center_index, feature_map
    else:
        return box_xy, box_wh, confidence, class_prob


import torch
import cv2
from torchvision.transforms.functional import to_tensor

from YOLOv4.anchor import get_anchor
from draw import Draw
from utils.nms import diou_nms
from utils.tools import reverse_letter_box, letter_box


def meshgrid(size, B, device):
    x = torch.arange(start=0, end=size[1], dtype=torch.float32, device=device)
    y = torch.arange(start=0, end=size[0], dtype=torch.float32, device=device)
    x, y = torch.meshgrid([x, y], indexing="xy")
    xy_grid = torch.stack(tensors=(y, x), dim=-1)
    xy_grid = torch.unsqueeze(xy_grid, dim=2)
    xy_grid = torch.unsqueeze(xy_grid, dim=0)
    xy_grid = xy_grid.repeat(B, 1, 1, 3, 1)
    return xy_grid


def encode_outputs(cfg, feature, feature_index):
    """
    编码YOLOv4的输出，生成用于loss计算的特征值
    :param feature: Tensor, shape: (batch_size, H, W, 3, C+5)
    :param feature_index: Int
    :return: Tensor, shape: (batch_size, feature_map_size, feature_map_size, 3, num_classes + 5)
    """
    anchors = get_anchor(cfg)

    shape = feature.size()
    dx_dy, dw_dh, conf, prob = torch.split(feature, [2, 2, 1, cfg["Model"]["num_classes"]], -1)

    xy_grid = meshgrid(size=shape[1:3], B=shape[0], device=cfg["device"])

    pred_xy = (torch.sigmoid(dx_dy) + xy_grid) / shape[1]
    pred_wh = torch.exp(dw_dh) * anchors[feature_index]
    pred_xywh = torch.cat(tensors=(pred_xy, pred_wh), dim=-1)
    pred_conf = torch.sigmoid(conf)
    pred_prob = torch.sigmoid(prob)
    pred_bbox = torch.cat(tensors=(pred_xywh, pred_conf, pred_prob), dim=-1)

    return pred_bbox


class Decode:
    def __init__(self, cfg, image_size):
        self.cfg = cfg
        self.device = cfg["device"]
        self.image_h = image_size[0]
        self.image_w = image_size[1]

        self.num_classes = cfg["Model"]["num_classes"]
        self.strides = cfg["Model"]["yolo_strides"]
        self.anchors = get_anchor(cfg)
        self.input_size = cfg["Train"]["input_size"]
        self.score_threshold = cfg["Nms"]["score_threshold"]
        self.iou_threshold = cfg["Nms"]["iou_threshold"]

    def __call__(self, outputs):
        boxes = list()
        for i, feature in enumerate(outputs):
            feature = torch.permute(feature, dims=(0, 2, 3, 1))
            shape = feature.size()
            feature = torch.reshape(feature, (shape[0], shape[1], shape[2], 3, -1))
            raw_xywh, raw_conf, raw_prob = torch.split(feature, [4, 1, self.num_classes], -1)
            pred_conf = torch.sigmoid(raw_conf)
            pred_prob = torch.sigmoid(raw_prob)

            raw_dxdy = raw_xywh[..., 0:2]
            raw_dwdh = raw_xywh[..., 2:4]
            xy_grid = meshgrid(size=shape[1:3], B=1, device=self.device)
            pred_xy = (torch.sigmoid(raw_dxdy) + xy_grid) / shape[1]
            pred_wh = torch.exp(raw_dwdh) * self.anchors[i]
            boxes.append(torch.cat(tensors=(pred_xy, pred_wh, pred_conf, pred_prob), dim=-1))

        boxes_tensor = torch.cat(tensors=[torch.reshape(box, (-1, 5 + self.num_classes)) for box in boxes], dim=0)
        boxes_tensor, scores_tensor, classes_tensor = self._filter(boxes_tensor)
        indices = diou_nms(boxes=boxes_tensor, scores=scores_tensor, iou_threshold=self.iou_threshold)
        return boxes_tensor[indices], scores_tensor[indices], classes_tensor[indices]

    def _filter(self, boxes):
        pred_xywh = boxes[:, 0:4]
        pred_conf = boxes[:, 4]
        pred_prob = boxes[:, 5:]

        # 将预测框的大小映射到原图尺寸
        pred_xyxy = reverse_letter_box(h=self.image_h, w=self.image_w, input_size=self.input_size, boxes=pred_xywh)
        # 遮住那些不符合条件的预测框
        pred_xyxy = torch.cat(
            tensors=(torch.max(pred_xyxy[:, :2], torch.tensor(data=(0, 0), dtype=torch.float32, device=self.device)),
                     torch.min(pred_xyxy[:, 2:],
                               torch.tensor(data=(self.image_w - 1, self.image_h - 1), dtype=torch.float32,
                                            device=self.device))),
            dim=-1)
        invalid_mask = torch.logical_or(torch.gt(pred_xyxy[:, 0], pred_xyxy[:, 2]),
                                        torch.gt(pred_xyxy[:, 1], pred_xyxy[:, 3]))
        pred_xyxy[invalid_mask] = 0

        bbox_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        area_mask = torch.gt(bbox_area, 0)

        classes = torch.argmax(pred_prob, dim=-1)
        scores = pred_conf * pred_prob[torch.arange(len(pred_xyxy)), classes]
        score_mask = torch.gt(scores, self.score_threshold)

        mask = torch.logical_and(area_mask, score_mask)
        boxes_tensor, scores_tensor, classes_tensor = pred_xyxy[mask], scores[mask], classes[mask]

        return boxes_tensor, scores_tensor, classes_tensor


def test_pipeline(cfg, model, image_path, save_dir=None, print_on=True, save_result=True):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    image, _, _ = letter_box(image, (cfg["Train"]["input_size"], cfg["Train"]["input_size"]))
    image = to_tensor(image)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device=cfg["device"])
    with torch.no_grad():
        outputs = model(image)
        decoder = Decode(cfg, image_size=(h, w))
        boxes, scores, classes = decoder(outputs)
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    classes = classes.cpu().numpy()
    if print_on:
        print("检测出{}个边界框，分别是：".format(boxes.shape[0]))
        print("boxes: ", boxes)
        print("scores: ", scores)
        print("classes: ", classes)

    painter = Draw(cfg)
    image_with_boxes = painter.draw_boxes_on_image(image_path, boxes, scores, classes)

    if save_result:
        # 保存检测结果
        cv2.imwrite(save_dir, image_with_boxes)
    else:
        return image_with_boxes
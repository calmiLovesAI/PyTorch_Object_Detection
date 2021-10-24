import torch
import cv2

from YOLOv3.anchor import get_anchor
from YOLOv3.nms import apply_nms
from draw import draw_boxes_on_image
from utils import letter_box, reverse_letter_box
from torchvision.transforms.functional import to_tensor


def generate_grid_index(length, device):
    x = torch.arange(start=0, end=length, step=1, dtype=torch.float32, device=device)
    y = torch.arange(start=0, end=length, step=1, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X = torch.reshape(X, shape=(-1, 1))
    Y = torch.reshape(Y, shape=(-1, 1))
    return torch.cat((X, Y), dim=-1)


def predict_bounding_bbox(cfg, feature_map, anchors, device, is_training=False):
    num_classes = cfg["Model"]["num_classes"]
    N, C, H, W = feature_map.size()
    feature_map = torch.permute(feature_map, dims=(0, 2, 3, 1))
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
    anchors = torch.tile(anchors, dims=[area, 1])
    box_wh = anchors * torch.exp(tw_th)

    # reshape
    center_index = torch.reshape(center_index, shape=(-1, H, W, 3, 2))
    box_xy = torch.reshape(box_xy, shape=(-1, H, W, 3, 2))
    box_wh = torch.reshape(box_wh, shape=(-1, H, W, 3, 2))
    feature_map = torch.reshape(feature_map, shape=(-1, H, W, 3, num_classes + 5))

    if is_training:
        return box_xy, box_wh, center_index, feature_map
    else:
        return box_xy, box_wh, confidence, class_prob


class Inference:
    def __init__(self, cfg, outputs, input_image_shape, device):
        self.cfg = cfg
        self.device = device
        self.outputs = outputs
        self.input_image_h = input_image_shape[0]
        self.input_image_w = input_image_shape[1]

    def _yolo_post_process(self, feature, scale_type):
        box_xy, box_wh, confidence, class_prob = predict_bounding_bbox(self.cfg, feature,
                                                                       get_anchor(self.cfg, scale_type, self.device),
                                                                       self.device, is_training=False)
        boxes = reverse_letter_box(self.input_image_h, self.input_image_w, self.cfg["Train"]["input_size"],
                                   torch.cat((box_xy, box_wh), dim=-1))
        boxes = torch.reshape(boxes, shape=(-1, 4))
        boxes_scores = confidence * class_prob
        boxes_scores = torch.reshape(boxes_scores, shape=(-1, self.cfg["Model"]["num_classes"]))
        return boxes, boxes_scores

    def get_results(self):
        boxes_list = list()
        boxes_scores_list = list()
        for i in range(3):
            boxes, boxes_scores = self._yolo_post_process(feature=self.outputs[i],
                                                          scale_type=i)
            boxes_list.append(boxes)
            boxes_scores_list.append(boxes_scores)
        boxes = torch.cat(boxes_list, dim=0)
        scores = torch.cat(boxes_scores_list, dim=0)
        return apply_nms(self.cfg, boxes, scores, self.device)


def test_pipeline(cfg, model, image_path, device, save_dir):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    image, _, _ = letter_box(image, (cfg["Train"]["input_size"], cfg["Train"]["input_size"]))
    image = to_tensor(image)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device=device)
    outputs = model(image)
    boxes, scores, classes = Inference(cfg=cfg, outputs=outputs, input_image_shape=(h, w), device=device).get_results()
    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    classes = classes.cpu().detach().numpy()

    image_with_boxes = draw_boxes_on_image(cfg, image_path, boxes, scores, classes)

    # 保存检测结果
    cv2.imwrite(save_dir, image_with_boxes)

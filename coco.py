import numpy as np
import torch
import os
import cv2

import yaml
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class Coco(Dataset):
    def __init__(self, cfg, transform=None):
        super(Coco, self).__init__()
        self.transform = transform

        coco_root = cfg["root"]
        self.images_root = os.path.join(coco_root, "train2017")
        anno_file = os.path.join(coco_root, "annotations", "instances_train2017.json")
        self.coco = COCO(annotation_file=anno_file)
        self.ids = list(self.coco.imgToAnns.keys())  # 图片id列表
        class_to_id = dict(zip(cfg["classes"], range(cfg["num_classes"])))
        class_to_coco_id = Coco._get_class_to_coco_id(self.coco.dataset["categories"])
        self.coco_id_to_class_id = dict([
            (class_to_coco_id[cls], class_to_id[cls])
            for cls in cfg["classes"]
        ])

    def __len__(self):
        """
        :return: 数据集长度
        """
        return len(self.ids)

    def __getitem__(self, item):
        img_id = self.ids[item]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        image_path = os.path.join(self.images_root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(image_path), '图片({})不存在'.format(image_path)
        # 读取图片
        image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 获取真实bbox
        target = self._get_true_bbox(target)
        target = np.array(target, dtype=np.float32)
        target = np.reshape(target, (-1, 5))
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    @staticmethod
    def _get_class_to_coco_id(categories):
        class_to_coco_id = dict()
        for category in categories:
            class_to_coco_id[category["name"]] = category["id"]
        return class_to_coco_id

    def _get_true_bbox(self, target):
        bboxes = list()
        for obj in target:
            # (xmin, ymin, w, h)格式
            bbox = obj["bbox"]
            # 转为(xmin, ymin, xmax, ymax)格式
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            class_idx = self.coco_id_to_class_id[obj["category_id"]]
            bboxes.append([*bbox, class_idx])
        return bboxes


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    with open(file="experiments/yolov3.yaml") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    coco_dataset = Coco(cfg["COCO"])

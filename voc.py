import os

import numpy as np
import torch
import cv2
import xml.dom.minidom as xdom
from torch.utils.data import Dataset


class Voc(Dataset):
    def __init__(self, cfg, transform=None):
        super(Voc, self).__init__()
        self.transform = transform
        self.xmls_path = cfg["root"] + "Annotations"
        self.images_path = cfg["root"] + "JPEGImages"
        self.xmls = os.listdir(self.xmls_path)
        self.class_index = dict((v, k) for k, v in enumerate(cfg["classes"]))

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, item):
        xml_file = os.path.join(self.xmls_path, self.xmls[item])
        image_path, target = self._parse_xml(xml_file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = np.array(target, dtype=np.float32)
        target = np.reshape(target, (-1, 5))
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def _parse_xml(self, xml):
        box_class_list = []
        DOMTree = xdom.parse(xml)
        annotation = DOMTree.documentElement
        image_name = annotation.getElementsByTagName("filename")[0].childNodes[0].data
        image_path = os.path.join(self.images_path, image_name)

        obj = annotation.getElementsByTagName("object")
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            bbox = o.getElementsByTagName("bndbox")[0]
            xmin = bbox.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = bbox.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = bbox.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = bbox.getElementsByTagName("ymax")[0].childNodes[0].data
            o_list.append(float(xmin))
            o_list.append(float(ymin))
            o_list.append(float(xmax))
            o_list.append(float(ymax))
            o_list.append(self.class_index[obj_name])
            box_class_list.append(o_list)
        return image_path, box_class_list

    def gather_wh(self):
        pass



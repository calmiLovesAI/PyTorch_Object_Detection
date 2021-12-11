import math
import cv2

from utils.tools import find_class_name


class Draw:
    def __init__(self, cfg=None):
        self.cfg = cfg
        # r, g, b
        self.colors = {
            "粉红": (255, 192, 203),
            "红色": (255, 0, 0),
            "紫罗兰": (238, 130, 238),
            "洋红": (255, 0, 255),
            "深天蓝": (0, 191, 255),
            "青色": (0, 255, 255),
            "春天的绿色": (60, 179, 113),
            "浅海洋绿": (32, 178, 170),
            "米色": (245, 245, 220),
            "小麦色": (245, 222, 179),
            "棕色": (165, 42, 42),
            "深灰色": (169, 169, 169),
            "黄色": (255, 255, 255),
            "紫红色": (255, 0, 255)
        }
        self.colors_list = [v for k, v in self.colors.items()]

    def _get_rgb_color(self, idx):
        return self.colors_list[idx % len(self.colors_list)]

    @staticmethod
    def _get_adaptive_zoom_ratio(h, w):
        """

        :param h: 图片的高
        :param w: 图片的宽
        :return: 边界框的标签与边界框的距离大小，标签文字大小
        """
        d = min(h, w) / 30
        r = min(h, w) / 1000
        r = math.ceil(r) * 0.5
        return d, r

    def draw_boxes_on_image(self, image_path, boxes, scores, classes):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        d, r = self._get_adaptive_zoom_ratio(h, w)
        boxes = boxes.astype(int)

        num_boxes = boxes.shape[0]
        for i in range(num_boxes):
            if self.cfg is not None:
                class_and_score = str(find_class_name(self.cfg, classes[i])) + ": {:.2f}".format(scores[i])
                # 获取类别对应的颜色
                bbox_color = self._get_rgb_color(classes[i])
            else:
                class_and_score = classes[i][0] + ": {:.2f}".format(scores[i])
                # 获取类别对应的颜色
                bbox_color = self._get_rgb_color(classes[i][1])
            bbox_color_bgr = bbox_color[::-1]
            cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]),
                          color=bbox_color_bgr,
                          thickness=2)
            cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - int(d)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=r, color=(0, 255, 255), thickness=1)
        return image

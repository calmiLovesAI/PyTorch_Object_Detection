import math
import cv2
import numpy as np


class Draw:
    def __init__(self):
        self.colors = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.314, 0.717, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32).reshape(-1, 3)
        # # r, g, b
        # self.colors = {
        #     "粉红": (255, 192, 203),
        #     "红色": (255, 0, 0),
        #     "紫罗兰": (238, 130, 238),
        #     "洋红": (255, 0, 255),
        #     "深天蓝": (0, 191, 255),
        #     "青色": (0, 255, 255),
        #     "春天的绿色": (60, 179, 113),
        #     "浅海洋绿": (32, 178, 170),
        #     "米色": (245, 245, 220),
        #     "小麦色": (245, 222, 179),
        #     "棕色": (165, 42, 42),
        #     "深灰色": (169, 169, 169),
        #     "黄色": (255, 255, 255),
        #     "紫红色": (255, 0, 255)
        # }
        # self.colors_list = [v for k, v in self.colors.items()]

    # def _get_rgb_color(self, idx):
    #     return self.colors_list[idx % len(self.colors_list)]
    #
    # @staticmethod
    # def _get_adaptive_zoom_ratio(h, w):
    #     """
    #
    #     :param h: 图片的高
    #     :param w: 图片的宽
    #     :return: 边界框的标签与边界框的距离大小，标签文字大小
    #     """
    #     d = min(h, w) / 30
    #     r = min(h, w) / 1000
    #     r = math.ceil(r) * 0.5
    #     return d, r

    def draw_boxes_on_image(self, image_path, boxes, scores, class_ids, class_names):
        image = cv2.imread(image_path)
        # h, w, _ = image.shape
        # d, r = self._get_adaptive_zoom_ratio(h, w)
        boxes = boxes.astype(int)

        num_boxes = boxes.shape[0]
        for i in range(num_boxes):
            print(class_ids)
            cls_id = class_ids[i]
            score = scores[i]
            x0 = boxes[i, 0]
            y0 = boxes[i, 1]
            x1 = boxes[i, 2]
            y1 = boxes[i, 3]
            color = (self.colors[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[i], score * 100)
            txt_color = (0, 0, 0) if np.mean(self.colors[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self.colors[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

            # class_and_score = classes[i][0] + ": {:.2f}".format(scores[i])
            # # 获取类别对应的颜色
            # bbox_color = self._get_rgb_color(classes[i][1])
            # bbox_color_bgr = bbox_color[::-1]
            # cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]),
            #               color=bbox_color_bgr,
            #               thickness=2)
            # cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - int(d)),
            #             fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=r, color=(0, 255, 255), thickness=1)
        return image

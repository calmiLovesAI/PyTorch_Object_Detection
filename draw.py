import cv2


def find_class_name(cfg, class_index):
    if cfg["Train"]["dataset_name"] == "voc":
        class_name_list = cfg["VOC"]["classes"]
        return class_name_list[class_index]
    if cfg["Train"]["dataset_name"] == "coco":
        class_name_list = cfg["COCO"]["classes"]
        return class_name_list[class_index]


def draw_boxes_on_image(cfg, image_path, boxes, scores, classes):
    image = cv2.imread(image_path)
    boxes = boxes.astype(int)

    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = str(find_class_name(cfg, classes[i])) + ": " + str(scores[i])
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(255, 0, 0), thickness=2)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 255), thickness=2)
    return image
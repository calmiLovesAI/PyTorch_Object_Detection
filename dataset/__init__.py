
dataset2cfg = {
    "voc": "VOC",
    "coco": "COCO",
    "custom": "Custom"
}


def find_class_name(cfg, class_index, keep_index=False):
    class_name_list = cfg[dataset2cfg[cfg["Train"]["dataset_name"]]]["classes"]
    if keep_index:
        return class_name_list[class_index], class_index
    return class_name_list[class_index]

import yaml


def get_dataset_classes():
    res = dict()
    with open(file="./experiments/VOC.yaml") as f:
        res["voc"] = yaml.load(f.read(), Loader=yaml.FullLoader)["classes"]
    with open(file="./experiments/COCO.yaml") as f:
        res["coco"] = yaml.load(f.read(), Loader=yaml.FullLoader)["classes"]
    with open(file="./experiments/Custom.yaml") as f:
        res["custom"] = yaml.load(f.read(), Loader=yaml.FullLoader)["classes"]
    return res


def find_class_name(dataset_name: str, class_index, keep_index=False):
    class_name_list = get_dataset_classes()[dataset_name.lower()]
    if keep_index:
        return class_name_list[class_index], class_index
    return class_name_list[class_index]

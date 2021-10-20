import yaml


with open(file="experiments/yolov3.yaml") as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

print(cfg)
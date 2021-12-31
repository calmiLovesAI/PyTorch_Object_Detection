from datetime import datetime

import torch

from load_yaml import load_yamls
from register import CenterNetCFG, YOLOv3CFG, SSDCFG, YOLOxSCFG, YOLOxMCFG, YOLOxLCFG, YOLOxXCFG


def get_time_format():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == '__main__':
    model = YOLOxSCFG

    model_name = model.name
    config = model.cfg_file
    test_pictures = [""]  # 测试图片的路径
    mode = "test"  # "train" or "test"
    model_filename = "yolox_s.pth"  # 模型文件名，位于路径"saved_model"下

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))
    cfg = load_yamls(model_yaml=config, device=device, model_name=model_name)

    print("Start {}ing {}.".format(mode, model_name))

    trainer = model.get_trainer(cfg)
    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test(images=test_pictures, prefix=get_time_format(), model_filename=model_filename, load_model=True)
    else:
        raise NotImplementedError("未实现的模式")

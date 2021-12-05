import torch

from load_yaml import load_yamls
from scripts import CenterNetTrainer

CONFIG = {
    "model_name": "centernet",
    "cfg": "centernet.yaml"
}

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))
    cfg = load_yamls(model_yaml=CONFIG["cfg"], device=device)

    if CONFIG["model_name"] == "centernet":
        CenterNetTrainer(cfg).train()

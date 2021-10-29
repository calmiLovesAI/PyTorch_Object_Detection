import torch


from YOLOv3.inference import test_pipeline
from YOLOv3.load_yaml import load_yaml
from YOLOv3.model import YoloV3
from video import Video

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    cfg = load_yaml()

    model = YoloV3(cfg["Model"]["num_classes"])
    model.to(device=device)
    model.load_state_dict(torch.load(cfg["Train"]["save_path"] + "YOLOv3.pth", map_location=device))
    model.eval()

    v = Video(cfg, model, device, cfg["Train"]["video_dir"], cfg["Train"]["video_save_dir"],
              pipeline_func=test_pipeline)
    if cfg["Train"]["save_video"]:
        v.write()
    else:
        v.show()

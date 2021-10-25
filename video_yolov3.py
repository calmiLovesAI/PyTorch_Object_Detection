import cv2
import torch
import yaml
import os

from YOLOv3.check import check_cfg
from YOLOv3.inference import test_pipeline
from YOLOv3.model import YoloV3


def detect_objects_in_video(cfg, device, model, video_dir):
    capture = cv2.VideoCapture(video_dir)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_cnt = 0
    while True:
        ret, frame = capture.read()
        if ret:
            frame_cnt += 1
            print("Processing frame {}".format(frame_cnt))
            tmp_dir = "./temp.jpg"
            cv2.imwrite(tmp_dir, frame)
            new_frame = test_pipeline(cfg, model, tmp_dir, device, print_on=False, save_result=False)
            cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("detect result", new_frame)
            cv2.waitKey(int(1000 / fps))
            os.remove(tmp_dir)
        else:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch version: {}, Device: {}".format(torch.__version__, device))

    with open(file="experiments/yolov3.yaml") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    check_cfg(cfg)

    model = YoloV3(cfg["Model"]["num_classes"])
    model.to(device=device)
    model.load_state_dict(torch.load(cfg["Train"]["save_path"] + "YOLOv3.pth", map_location=device))
    model.eval()

    detect_objects_in_video(cfg, device, model, video_dir=cfg["Train"]["video_dir"])


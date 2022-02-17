# SkeNetch
SkeNetch，即Sketch Network，是一个基于PyTorch的深度学习工具，专注于目标检测领域。<br>
SkeNetch, which means Sketch Network, is a deep learning tool based on PyTorch. SkeNetch focuses on the object detection.
# Algorithms that have been implemented.
| Algorithm | Pretrained models | Paper                             |
|-----------|-------------------|-----------------------------------|
| YOLOv3    |                   | https://arxiv.org/abs/1804.02767  |
| CenterNet | [VOC](https://github.com/calmisential/SkeNetch/releases/download/Weights/CenterNet_voc_epoch_200.pth)           | https://arxiv.org/abs/1904.07850  |
| SSD       |                   | https://arxiv.org/abs/1512.02325  |
| YOLOX | | https://arxiv.org/abs/2107.08430  |
# Quick start
1. Make sure that you have installed PyTorch 1.10.0 and torchvision 0.11.1 or higher.
2. Run the following to install dependencies.
```commandline
pip install -r requirements.txt
```
3. Install `pycocotools`.
4. Download [COCO2017](https://cocodataset.org/#download) and [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit), and then extract them under `data` folder, make them look like this:
```
|-- data
    |-- coco
    |   |-- annotations
    |   |   |-- instances_train2017.json
    |   |   `-- instances_val2017.json
    |   `-- images
    |       |-- train2017
    |       |   |-- ... 
    |       `-- val2017
    |           |-- ... 
    `-- VOCdevkit
        |
        `-- VOC2012
            |
            `-- |-- Annotations
                |-- ImageSets
                |-- JPEGImages
                |-- SegmentationClass
                `-- SegmentationObject
```
5. Modify the configuration file under the `experiments` folder according to your needs.
6. Change the `CONFIG` parameter in `launcher.py`, and then run `launcher.py` to start training or detect multiple pictures at once.
# Results
1. YoloV3 on VOC<br>
![测试图片1.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample1.jpg?raw=True)
![测试图片2.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample2.jpg?raw=true)
![测试图片3.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample3.jpg?raw=true)
![测试图片4.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample4.jpg?raw=true)

# Acknowledgements
- https://github.com/hunglc007/tensorflow-yolov4-tflite
- https://github.com/amdegroot/ssd.pytorch code for processing COCO dataset
- https://github.com/ultralytics/yolov5
- https://github.com/calmisential/YOLOv4_PyTorch
- https://github.com/calmisential/CenterNet_TensorFlow2
- https://github.com/xingyizhou/CenterNet
- https://github.com/amdegroot/ssd.pytorch
- https://www.bilibili.com/video/BV1vK411H771
- https://github.com/Megvii-BaseDetection/YOLOX

# SkeNetch
SkeNetch，即Sketch Network，是一个基于PyTorch的深度学习工具。 SkeNetch 实现了一些深度学习算法，帮助用户快速训练、测试和部署网络模型。<br>
SkeNetch, which means Sketch Network, is a deep learning tool based on PyTorch.
SkeNetch implements some deep learning algorithms to help users quickly train, test, and deploy network models.
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
    |-- VOCdevkit
    `-- |-- Annotations
        |-- ImageSets
        |-- JPEGImages
        |-- SegmentationClass
        |-- SegmentationObject
```
5. Modify the configuration file under the `experiments` folder according to your needs.
6. Change the `CONFIG` parameter in `launcher.py`, and then run `launcher.py` to start training or detect multiple pictures at once.
# Results
1. YoloV3 on VOC<br>
![测试图片1.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample1.jpg?raw=True)
![测试图片2.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample2.jpg?raw=true)
![测试图片3.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample3.jpg?raw=true)
![测试图片4.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample4.jpg?raw=true)
# TODO list
1. Training and predicting code
- [x] Yolo_v3
- [x] Yolo_v4
- [ ] Yolo_v5
- [ ] Yolo_X
- [x] CenterNet
2. Release models trained on COCO
- [ ] Yolo_v3
- [ ] Yolo_v4
- [ ] Yolo_v5
- [ ] Yolo_X
- [ ] CenterNet

# References
- https://github.com/hunglc007/tensorflow-yolov4-tflite
- https://github.com/amdegroot/ssd.pytorch code for processing COCO dataset
- https://github.com/ultralytics/yolov5
- https://github.com/calmisential/YOLOv4_PyTorch
- https://github.com/calmisential/CenterNet_TensorFlow2
- https://github.com/xingyizhou/CenterNet
- https://github.com/amdegroot/ssd.pytorch

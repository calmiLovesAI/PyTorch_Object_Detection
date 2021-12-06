# YOLO_Series
YOLOv3, YOLOv4, YOLOv5, YOLOX的PyTorch实现。（持续更新中......）

# Quick start
1. Make sure that you have installed PyTorch 1.10.0 and torchvision 0.11.1 or higher.
2. Run the following to install dependencies.
```commandline
pip install -r requirements.txt
```
3. Modify the configuration file under the `experiments` folder according to your needs.
4. Change the `CONFIG` parameter in `train.py`, and then run `train.py` to start training.
5. Change the `CONFIG` parameter in `test.py`, and then run `test.py` to detect multiple pictures at once.
# 运行结果(Results)
1. YoloV3 on VOC<br>
![测试图片1.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample1.jpg?raw=True)
![测试图片2.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample2.jpg?raw=true)
![测试图片3.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample3.jpg?raw=true)
![测试图片4.jpg](https://github.com/calmisential/YOLO_Series/blob/main/assets/yolov3_voc_sample4.jpg?raw=true)
# TODO list
1. 训练和测试代码
- [x] Yolo_v3
- [x] Yolo_v4
- [ ] Yolo_v5
- [ ] Yolo_X
- [ ] CenterNet
2. 发布在COCO数据集上训练好的模型
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

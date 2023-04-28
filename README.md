# Object Detection with YOLOv5
This repository contains an object detection projects using [YOLOv5](https://github.com/ultralytics/yolov5) on [Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) and [Urban Congestion Bird Eye View Image](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=241) datasets.

### Create virtual environment
```python
conda create -n <ENV_NAME> python = 3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Create datasets to train YOLOv5 object detection model
##### 1) Extract data from .xml or .json files:

```python

python datasets/make_dataset_urban.py
python datasets/make_dataset_road.py

```

##### 2) Check data to ensure that bounding boxes match with images well:

```python

python datasets/check_data_urban.py
python datasets/check_data_road.py

```

##### 3) Create directories and move images/bounding boxes to images and labels directories:

```python

python datasets/dataset_urban.py
python datasets/dataset_road.py

```

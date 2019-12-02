# Multi-type_vehicles_flow_statistics
According to YOLOv3 and SORT algorithms, counting multi-type vehicles. Implemented by Pytorch.

## Reference
- yolov3-darknet  https://github.com/pjreddie/darknet
- yolov3-pytorch  https://github.com/eriklindernoren/PyTorch-YOLOv3
- sort https://github.com/abewley/sort

## Dependencies
- ubuntu/windows
- cuda>=10.0
- python>=3.6
- `pip3 install -r requirements.txt`

## Usage
1. Download the pre-trained yolov3 weight file [here](https://pjreddie.com/media/files/yolov3.weights) and put it into `weights` directory;  
2. run `python3 app.py` ;
3. Select video and double click the image to select area.

## Demo
![avatar](https://github.com/wsh122333/Multi-type_vehicles_flow_statistics/raw/master/asserts/demo1.gif)

![avatar](https://github.com/wsh122333/Multi-type_vehicles_flow_statistics/raw/master/asserts/demo2.gif)


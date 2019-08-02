# YOLOV3_pytorch
 Pytorch implementation of YOLOv3
 
 add voc2007 support
 
 can train with voc2007 and eval.
 
 and can draw the result
 
 modify train.py  74 rows data_dir
 
 
$ mkdir weights

$ cd weights/

$ bash ../requirements/download_weights.sh

$ bash requirements/getcoco.sh



python train.py --weights_path weights/darknet53.conv.74

thanks to：

https://github.com/DeNA/PyTorch_YOLOv3

more detail please see the link above.

import os
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from utils.utils import *


class VOCDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, model_type, data_dir,
                 name='trainval', img_size=416,
                 augmentation=None, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        self.model_type = model_type
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(name))
        self.name=name
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True

        # load image and preprocess
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = cv2.imread(img_file)

        #图像大小变换以及填充
        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)

        if self.random_distort:
            #HSV变换
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        #水平翻转
        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))

        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            #读取标签
            name = obj.find('name').text.lower().strip()
            bndbox_anno = obj.find('bndbox')

            # subtract 1 to make pixel indexes 0-based
            bbox_=[int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
            x1 = float(bndbox_anno.find('xmin').text) - 1
            y1 = float(bndbox_anno.find('ymin').text) - 1
            x2 = float(bndbox_anno.find('xmax').text) - 1
            y2 = float(bndbox_anno.find('ymax').text) - 1

            bbox_[0] = x1
            bbox_[1] = y1
            bbox_[2] = x2-x1
            bbox_[3] = y2-y1

            #label为【x,y,w,h】 这里x,y是左上角坐标
            if bbox_[2] > self.min_size and bbox_[3] > self.min_size:
                labels.append([])
                labels[-1].append(VOC_BBOX_LABEL_NAMES.index(name))
                labels[-1].extend(bbox_)

        # yolo标签是50个bounding box，不够用零来凑
        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)
        return img, padded_labels, info_img, id_




VOC_BBOX_LABEL_NAMES = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor')

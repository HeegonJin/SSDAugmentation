import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import collections
from myaug import SSDAugmentation
import cv2
import glob
import os.path as osp
    
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class VOCAnnotationTransform(object):
    """    
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes    
    
    class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
        """
    def __init__(self, class_to_ind = None, keep_difficult = False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        #print(self.class_to_ind)
        
    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        root = target.getroot()
        res = []
        label = -1
        for obj in root.iter('object'):
                
            name = obj.find('name').text
            label = self.class_to_ind[name]
            
            
            for bbox in obj.iter("bndbox"):
                bndbox = ['xmin','ymin','xmax','ymax']
                bndbox[0] = int(float(bbox.find(bndbox[0]).text)) / width
                bndbox[1] = int(float(bbox.find(bndbox[1]).text)) / height
                bndbox[2] = int(float(bbox.find(bndbox[2]).text)) / width
                bndbox[3] = int(float(bbox.find(bndbox[3]).text)) / height
            bndbox.append(label)
            res += [bndbox]
        #print(res)
        return res
        
class VOCDetection(object):

    def __init__(self, root, image_sets = [('2007', 'trainval'), ('2012', 'trainval')],
             transform = None, target_transform = VOCAnnotationTransform()):
        self.root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform
        
        self.annopath = osp.join(root, '%s', 'Annotations', '%s.xml')
        self.imgpath = osp.join(root, '%s', 'JPEGImages', '%s.jpg')
        self.ids = []
        
        for year, name in self.image_sets:
            Idfile = open(osp.join(root, 'VOC'+year, 'ImageSets', 'Main', 'trainval.txt'),'r')
            while True:
                Id = Idfile.readline().strip()
                if not Id:
                    break
                self.ids.append(('VOC'+year, Id))
            Idfile.close()
        # print(self.ids)
                    
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)
    
    def pull_item(self, index):
        img_id = self.ids[index] # cheated here
        
        image = cv2.imread(self.imgpath % img_id)
        target = ET.parse(self.annopath % img_id)
        
        width, height, _ = image.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
            
        target = np.array(target)
        if self.transform is not None:
            image, boxes, labels = self.transform(image, target[:,:4], target[:, 4])
        image = image[:,:,(2,1,0)]
        targets = np.hstack((boxes, np.expand_dims(labels, axis = 1)))
        return torch.from_numpy(image).permute(2,0,1), targets, height, width
       
        
DATAROOT = "c:/users/user/data/VOCDEVKIT"
a = VOCDetection(DATAROOT,transform = SSDAugmentation())     
a.pull_item(6)

b = VOCAnnotationTransform()
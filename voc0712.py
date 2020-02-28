import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import collections
from augmentation import SSDAugmentation
import cv2
    
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class VOCAnnotationTransfrom(object):
    def __init__(self, class_to_ind = None):
        
      self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
      self.bbox_lst = []
      self.annotation = None
      self.filename = None
    def __call__(self,target):
      self.annotation = target
      anno = self.annotation['annotation']
      filename = anno['filename']
      size = anno['size']
      width = size["width"]
      height = size["height"]
      objects = []
      objects.append(anno['object'])
      if type(objects[0]) == dict:
          for obj in objects:
              box = []
              name = obj['name']
              bbox = obj['bndbox']
              label_idx = self.class_to_ind[name]
              box.append(int(bbox['xmin'])/int(width))
              box.append(int(bbox['ymin'])/int(height))
              box.append(int(bbox['xmax'])/int(width))
              box.append(int(bbox['ymax'])/int(height))
              box.append(int(label_idx))
              self.bbox_lst.append(box)
      else:
          for obj in objects:
              for i in range(len(obj)):
                  box = []
                  name = obj[i]['name']
                  bbox = obj[i]['bndbox']
                  label_idx = self.class_to_ind[name]
                  box.append(int(bbox['xmin'])/int(width))
                  box.append(int(bbox['ymin'])/int(height))
                  box.append(int(bbox['xmax'])/int(width))
                  box.append(int(bbox['ymax'])/int(height))
                  box.append(int(label_idx))
                  self.bbox_lst.append(box)
           
      self.bbox_lst = np.array(self.bbox_lst)
      self.filename = filename
      return self.bbox_lst
     
          
          
      
    
class VOCDetection(object):
    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def __init__(self, root, transforms, target_transfrom = None):
        self.root = root
        self.transforms = SSDAugmentation()
        self.target_transfrom = VOCAnnotationTransfrom()
        # load all image files, sorting them to
        
        voc_root = self.root
        image_set = "train"
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')
        
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.imgs = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annos = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
      
    def __getitem__(self, idx):
        # load images ad masks
        #img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        #anno_path = os.path.join(self.root, "Annotations", self.annos[idx])
        img = cv2.imread(self.imgs[idx],cv2.IMREAD_COLOR)
        target = self.parse_voc_xml(
            ET.parse(self.annos[idx]).getroot())
        
        if self.target_transfrom is not None:
            transform = VOCAnnotationTransfrom()
            target = transform(target)
            target = np.array(target)

        if self.transforms is not None:
            img, boxes, labels = self.transforms(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target

    def __len__(self):
        return len(self.imgs)
# import os
# import numpy as np
# import torch
# from PIL import Image
# import xml.etree.ElementTree as ET
# import collections
# from myaug import SSDAugmentation
# import cv2
# import glob
# import os.path as osp
    
# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

# class VOCAnnotationTransform(object):
#     """    
#     Transforms a VOC annotation into a Tensor of bbox coords and label index
#     Initilized with a dictionary lookup of classnames to indexes    
    
#     class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
#             (default: alphabetic indexing of VOC's 20 classes)
#         keep_difficult (bool, optional): keep difficult instances or not
#             (default: False)
#         height (int): height
#         width (int): width
#         """
#     def __init__(self, class_to_ind = None, keep_difficult = False):
#         self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
#         self.keep_difficult = keep_difficult
#         #print(self.class_to_ind)
        
#     def __call__(self, target, width, height):
#         """
#         Arguments:
#             target (annotation) : the target annotation to be made usable
#                 will be an ET.Element
#         Returns:
#             a list containing lists of bounding boxes  [bbox coords, class name]
#         """
#         res = []
#         for obj in target.iter('object'):
#             difficult = int(obj.find('difficult').text) == 1 
#             if not self.keep_difficult and difficult:
#                 continue 
#             name = obj.find('name').text.lower().strip() #소문자, 공백삭제
#             bbox = obj.find('bndbox')
            
#             pts = ['xmin', 'ymin', 'xmax', 'ymax']
#             bndbox = []
#             for i, pt in enumerate(pts):
#                 cur_pt = int(bbox.find(pt).text) - 1 # 1을 빼주는 이유는?
#                 # scale에 대한 상대적인 값으로 나타내줌
#                 cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
#                 bndbox.append(cur_pt)
#                 label_idx = self.class_to_ind[name]
#                 bndbox.append(label_idx)
#                 res += [bndbox] # [[xmin, ymin, xmax, ymax, label_ind]]
#         return res
            
# class VOCDetection(object):

#     def __init__(self, root, image_sets = [('2007', 'trainval'), ('2012', 'trainval')],
#               transform = None, target_transform = VOCAnnotationTransform()):
#         self.root = root # DATAROOT = "c:/users/user/data"
#         self.image_sets = image_sets
#         self.transform = transform
#         self.target_transform = target_transform
#         self.annopath = os.path.join('%s', 'Annotations', '%s.xml')
#         self.imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
#         # self.annopath % ('c:/users/user/data/VOCDEVKIT\\VOC2012', '2011_003276')
#         # -> c:/users/user/data/VOCDEVKIT\VOC2012\Annotations\2011_003276.xml
#         self.ids = []
    
#         for (year, name) in self.image_sets:
#             rootpath = os.path.join(self.root, 'VOC' + year)
#             for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
#                 self.ids.append((rootpath, line.strip()))
#         #print(self.annopath % ('c:/users/user/data/VOCDEVKIT\\VOC2012', '2011_003276'))
#     # anno_file_list = os.listdir(anno_root)
#     # self.anno_list = [file for file in anno_file_list if file.endswith(".xml")]
#     # image_root = os.path.join(root, "JPEGimages")
#     # image_file_list = os.listdir(image_root)
#     # self.image_list = [file for file in image_file_list if file.endswith(".jpg")]
#     # print(self.anno_list)
#     # print(self.image_list)
        
        
#     def __getitem__(self, index):
#         im, gt, h, w = self.pull_item(index)

#         return im, gt

#     def __len__(self):
#         return len(self.ids)
    
#     def pull_item(self, index):
#         img_id = self.ids[index]
#         target = ET.parse(self.annopath % img_id).getroot()
#         img = cv2.imread(self.imgpath % img_id)
#         height, width, channels = img.shape

#         if self.target_transform is not None:
#             target = self.target_transform(target, width, height)
        
#         if self.transform is not None:
#             target = np.array(target)
#             img, boxes, labels = self.transform(img, target[:,:4], target[:,4])
#             #to rgb
#             img = img[:,:,(2,1,0)]
#             #img = img.transpose(2,0,1)
#             #print(img.shape)
#             target = np.hstack((boxes, np.expand_dims(labels, axis = 1)))
#             # boxes가 여러 개일 때 (n*(1*4), 2-dim), labels(1*n, 0-dim)을 (n*1, 2-dim)에 맞추고, 각각 한 줄로 합친다.
#         return torch.from_numpy(img).permute(2, 0, 1), target, height, width
#             #NumPy 배열을 텐서로 바꿀 때는 torch.from_numpy() 함수를 쓸 수 있다.
#             # 위에서 img.transpose 했다면,return torch.from_numpy(img), target, height, width 
#             # transpose, permute 해주는 이유는 H,W,C 를 C,H,W 로 바꿔줘야 하므로
#             # tennsors in pytorch are formatted in CHW(BCHW) by default,
# DATAROOT = "c:/users/user/data/VOCDEVKIT"
# a = VOCDetection(DATAROOT,transform = SSDAugmentation())     
# a.pull_item(0)

# b = VOCAnnotationTransform()
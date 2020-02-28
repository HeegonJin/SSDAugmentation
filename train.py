
# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import os
import torch
import torchvision
import sys
from XMLloader import XMLloader
import voc0712
from voc0712 import VOCDetection
from voc0712 import VOCAnnotationTransfrom
from augmentation import SSDAugmentation


DATAROOT = "c:/users/user/data"
VOCROOT = os.path.join(DATAROOT, "VOCdevkit", "VOC2007")
ANNOROOT = os.path.join(DATAROOT, "VOCdevkit", "VOC2007", "Annotations","000001.xml")


dataset = VOCDetection(VOCROOT,transforms = SSDAugmentation())
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=32)

batch_iterator = iter(data_loader)
images, targets = next(batch_iterator)
print(images, targets)
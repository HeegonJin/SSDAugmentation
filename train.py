
# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import os
import torch
import torchvision
import sys
import voc0712
from myvoc import VOCDetection
from myaug import SSDAugmentation
from torchvision import transforms
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


DATAROOT = "c:/users/user/data/VOCDevkit"
dataset = VOCDetection(DATAROOT, transform = SSDAugmentation())
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=32, shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

batch_iterator = iter(data_loader)
images, targets = next(batch_iterator)
print(images,targets)
img = transforms.ToPILImage()(images[0])
img.show()
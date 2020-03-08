# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:20:37 2020

@author: User
"""
import numpy as np
import numpy.random as random
import cv2

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    
    
    def __call__(self,img, boxes, labels):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels
    
class ConvertFromInts(object):
    def __call__(self, image, boxes = None, labels = None):
        return image.astype(np.float32), boxes, labels
    
            
class ToAbsoluteCoords(object):
    def __call__(self, image, boxes = None, labels = None):
        height, width, channnels = image.shape
        boxes[:,0] *= width 
        boxes[:,2] *= width
        boxes[:,1] *= height
        boxes[:,3] *= height
        
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower = 0.5, upper = 1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0
        
    def __call__(self, image, boxes = None, labels = None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current ="BGR", transform = "HSV"):
        self.current = current
        self.transform = transform
        
    def __call__(self,image, boxes = None, labels = None):
        if self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
            
        return image, boxes, labels
    
class RandomSaturation(object):  # HSV형태라고 가정
    def __init__(self, lower = 0.5, upper = 1.5):
        self.lower = lower
        self.upper = upper
        
        assert self.upper >= self.lower
        assert self.lower >= 0

    def __call__(self, image, boxes = None, labels = None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower,self.upper) 
            
        return image, boxes, labels
    
class RandomHue(object): # H 값은 0°~360°의 범위
    def __init__(self, delta = 18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
    
    def __call__(self, image, boxes = None, labels = None):
        if random.randint(2):
            #+-delta 범위의 값을 H값에 더해주고, 360 보다 크면 360 뺴주고 0보다 작으면 360 더해줌
            image[:,:,0] += random.uniform(-self.delta, self.delta)
            image[:,:,0][image[:,:,0]>=360.0] -= 360.0
            image[:,:,0][image[:,:,0]<0.0] += 360.0
        
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta = 32.0):
        assert delta >=0.0 and delta <= 255.0
        self.delta = delta
        
    def __call__(self, image, boxes = None, labels = None):
        if random.randint(2):
            image += random.uniform(-self.delta, self.delta)
            
        return image, boxes, labels        
                
class SwapChannels(object): #넘파이에서는 튜플 형식으로 인덱스 접근, 배열의 순서 바꿀수 있음
    def __init__(self, swaps):
        self.swaps = swaps
    
    def __call__(self,image):
        image = image[:,:,self.swaps]
        return image
    
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = [(0,1,2), (0,2,1), (1,0,2), (1,2,0),
                      (2,0,1),(2,1,0)]
    
    def __call__(self, image, boxes = None, labels = None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        
        return image, boxes, labels
        
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform = "HSV"),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current = "HSV", transform = "BGR"),
            RandomContrast()
            ]
        self.random_brightness = RandomBrightness()
        self.random_light_noise = RandomLightingNoise()
    
    def __call__(self, image, boxes = None, labels = None):
        im = image.copy()
        im, boxes, labels = self.random_brightness(im, boxes, labels)
        execute = random.randint(0,1)
        if execute == 0:
            pd = self.pd[0:5]
        else:
            pd = self.pd[2:]
        distort = Compose(pd)
        im, boxes, labels = distort(image, boxes, labels)
        return self.random_light_noise(im, boxes, labels)
        
# argument : img, boxes, lables 
#           np.array(float32) 300 300 3 height width channels
#           boxes n * (1 * 4) widthmin widthmax heightmin heightmax


class Expand(object):
    def __init__(self, mean):
        self.mean = mean
        
    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels
        else:
            ratio = random.randint(1,4)
            height, width, depth = image.shape
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)
            
            expand_image =\
                np.zeros((int(height * ratio), int(width * ratio), depth),
                     dtype = image.dtype)
            expand_image[:,:,:] = self.mean
            expand_image[int(top):int(top+height),int(left):int(left+width), :] = image
            image = expand_image
            
            boxes = boxes.copy()
            boxes[:,:2] = (int(left), int(top))
            boxes[:, 2:] = (int(left), int(top))
            
            return image, boxes, labels

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min = 0, a_max = np.inf)
    return inter[:, 0] * inter[:,1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union 
        

class RandomSampleCrop(object):

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
    def __call__(self, image, boxes = None, labels = None):
        height, width, _ = image.shape
        while True:
            
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels
                
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
                
            for _ in range(50):
                current_image = image
                
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                
                #ratio constraint from 0.5 to 2
                if h / w < 0.5 or h / w > 2:
                    continue
                
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                
                # convert to integer rect x1, y1, x2, y2
                rect = np.array([int(left), int(top), int(left+width), int(top + height)])
                
                # calculate IoU (jaccard overlay) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)
                
                # is min and max overlap constraint satisfied? if not, try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]
                
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                
                # mask in all gt boxes that outside of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                
                mask = m1 * m2
                
                # have any valid boxes? if not, try again
                if not mask.any():
                    continue
                
                current_boxes = boxes[mask, :].copy()
                
                current_labels = labels[mask]
                
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[2:]
                
                return current_image, current_boxes, current_labels
                
                
class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:,::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes
    
class ToPercentCoords(object):
    def __call__(self, image, boxes = None, labels = None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        
        return image, boxes, labels


class Resize(object):
    def __init__(self, size = 300):
        self.size = size
    
    def __call__(self, image, boxes = None, labels = None):
        image = cv2.resize(image, (self.size, self.size))

        return image, boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype = np.float32)
    
    def __call__(self, image, boxes = None, labels = None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels
    
        
    
    
class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
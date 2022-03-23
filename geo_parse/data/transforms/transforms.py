# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target_det=None, target_seg=None):
        for t in self.transforms:
            image, target_det, target_seg = t(image, target_det, target_seg)
        return image, target_det, target_seg

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Resize(object):
    '''
        Resize the training samples, and at the same time resize the corresponding target.
    '''
    def __init__(self, min_size, max_size, fpn_strides):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.fpn_stride = fpn_strides

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))

            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target_det=None, target_seg=None):

        size = self.get_size(image.size)
        image = F.resize(image, size)
        # The sacle of segmentation map of target is reduced to 1.0/self.fpn_stride
        # for decreasing the computation
        seg_size = tuple(round(item/self.fpn_stride) for item in image.size)

        if isinstance(target_det, list) and isinstance(target_seg, list):
            target_det = [t.resize(image.size) for t in target_det]
            target_seg = [t.resize(seg_size) for t in target_seg]
        elif target_det is None and target_seg is None:
            return image
        else:
            target_det = target_det.resize(image.size)
            target_seg = target_seg.resize(seg_size)

        return image, target_det, target_seg

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target_det, target_seg):
        if random.random() < self.prob:
            flip_method = random.choice([0,1,2])
            if flip_method==0:
                image = F.hflip(image)
                target_det = target_det.transpose(0)
                target_seg = target_seg.transpose(0)
            elif flip_method==1:
                image = F.vflip(image)
                target_det = target_det.transpose(1)  
                target_seg = target_seg.transpose(1)   
            elif flip_method==2:
                image = F.vflip(F.hflip(image))
                target_det = target_det.transpose(0).transpose(1)
                target_seg = target_seg.transpose(0).transpose(1)  
        return image, target_det, target_seg

class ToTensor(object):
    def __call__(self, image, target_det=None, target_seg=None):

        return F.to_tensor(image), target_det, target_seg

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target_det=None, target_seg=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target_det is None and target_seg is None:
            return image
        return image, target_det, target_seg

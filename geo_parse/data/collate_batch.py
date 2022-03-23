# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from geo_parse.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader

    GT includes targets_det, targets_seg and targets_rel corresponding to three sub-tasks of PGDP:
    identifying and locating non-geometric and geometric primitives, discovering relationships among them.
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets_det = transposed_batch[1]
        targets_seg = transposed_batch[2]
        targets_rel = transposed_batch[3]
        img_ids = transposed_batch[4]
        return images, targets_det, targets_seg, targets_rel, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

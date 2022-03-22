# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import os
import numpy as np
import json
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from geo_parse.modeling.detector import build_detection_model
from geo_parse.utils.checkpoint import DetectronCheckpointer
from geo_parse.structures.image_list import to_image_list
from geo_parse.data.datasets.evaluation.geo.generate_relation_GAT import get_relation_GAT
from geo_parse.data.datasets.evaluation.geo.generate_logic_form import get_logic_form
from geo_parse.data.datasets.evaluation.geo.utils import show_img_result_demo
from geo_parse.structures.bounding_box import BoxList

class Resize(object):
    def __init__(self, min_size, max_size, fpn_strides):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.fpn_stride = fpn_strides
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

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

class GeoDemo(object):

    def __init__(
        self,
        cfg,
        categories = None,
        text_content_path = None,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.cpu_device = torch.device("cpu")
        self.device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else self.cpu_device)
        self.model.to(self.device)
        self.categories = categories
        with open(text_content_path, 'rb') as file:
            self.text_contents = json.load(file)
        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.transforms = self.build_transform()

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size, fpn_strides=cfg.MODEL.SEG.FPN_STRIDES),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform
    
    def get_target_det(self, annot_each, img_size):

        boxes, classes, text_contents, classes_text = [], [], [], []

        for obj in annot_each['symbols']:
            if obj["sym_class"]!='text' and obj["sym_class"]!='arrow':
                boxes.append(obj["bbox"])
                text_contents.append(obj["text_content"])
            elif obj["text_class"]!='others':
                boxes.append(obj["bbox"])
                text_contents.append(obj["text_content"])
      
        boxes = torch.as_tensor(boxes).reshape(-1, 4) 
        target_det = BoxList(boxes, img_size, mode="xywh").convert("xyxy")
        target_det.add_field("labels", torch.tensor(classes))
        target_det.add_field("text_contents", text_contents)
        target_det.add_field("labels_text", classes_text)
        return  target_det

    def run_on_opencv_image(self, img_path):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        """
        img_name = os.path.basename(img_path)
        img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),1)
        if img is None: return
        for type_img in ['.png', '.jpg', '.JPEG', '.PNG']: 
            if type_img in img_name:
                img_name_pre = img_name[:img_name.find(type_img)]
                break

        prediction = self.compute_prediction(img)

        img_size = (self.text_contents[img_name_pre]['width'], self.text_contents[img_name_pre]['height'])
        gt_boxlist = self.get_target_det(self.text_contents[img_name_pre], img_size)

        relation_each, elem_dict = get_relation_GAT(prediction, img_name, gt_boxlist, self.cfg)
        logic_form_each, id2name_dict = get_logic_form(relation_each, elem_dict, img_name)

        sym_img, point_img, line_img, circle_img = show_img_result_demo(img, self.categories['geo'], 
                                                                    prediction, id2name_dict=id2name_dict)
        return sym_img, point_img, line_img, circle_img, relation_each, logic_form_each

    def compute_prediction(self, original_image):
        """
            Arguments:
                original_image (np.ndarray): an image as returned by OpenCV
            Returns:
                BoxList: the symbol detected objects.
                GeoList: the geometric detected objects
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        # always single image is passed at a time
        prediction = predictions[0]
        prediction['det']=prediction['det'].to(self.cpu_device)
        # reshape prediction into the original image size
        height, width = original_image.shape[:-1]
        prediction['det'] = prediction['det'].resize((width, height))
        prediction['seg'] = prediction['seg'].resize((width, height))

        return prediction


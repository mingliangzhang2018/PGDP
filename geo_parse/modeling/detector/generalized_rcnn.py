# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from geo_parse.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..geo_seg.geo_seg import build_seg
from ..geo_visemb.geo_visemb import build_visemb
from ..geo_rel.geo_rel import build_rel
import torch.nn.functional as F


class GeneralizedRCNN(nn.Module):
    
    """
        Main class for Generalized R-CNN. Currently supports boxes, masks and relation.
        It consists of five main parts:
        - backbone
        - rpn (fcos)
        - seg
        - visemb
        - rel
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        # just for building head and loss
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # for instance segmentation module
        self.seg = build_seg(cfg, self.backbone.out_channels)
        # visual embedding
        self.visemb = build_visemb(cfg, self.backbone.out_channels)
        # relation module
        self.rel = build_rel(cfg)

        w = h = int(cfg.INPUT.MAX_SIZE_TRAIN/cfg.MODEL.SEG.FPN_STRIDES)+50
        xm = torch.linspace(0, 1.0, w).view(1, 1, -1).expand(1, h, w)
        ym = torch.linspace(0, 1.0, h).view(1, -1, 1).expand(1, h, w)
        self.location_map = torch.cat((xm, ym), 0)

    def forward(self, images, targets_det=None, targets_seg=None, targets_rel=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets_det (list[BoxList]): ground-truth boxes present in the image (optional)
            targets_seg (list[GeoList]): ground-truth masks present in the image (optional)
            targets_rel (list[RelList]): ground-truth relation map present in the image (optional)

        Returns:
            result (proposals or losses): the output from the model.
                    During training, it returns a dict[Tensor] which contains the losses.
                    During testing, it returns dict[list[BoxList], list[GeoList], list[RelList]] contains additional fields
        """
        if self.training and (targets_det is None or targets_seg is None or targets_rel is None):
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors) 
        
        proposals_det, proposal_det_losses = self.rpn(images, features[1:], targets_det)

        loc_map = self.location_map[:,:features[0].shape[2],:features[0].shape[3]].contiguous()
        loc_map = loc_map.expand(len(features[0]),-1,-1,-1).cuda()
        
        features_share = torch.cat((features[0], loc_map), 1)

        proposals_seg, proposal_seg_losses = self.seg(images, features_share, targets_seg)

        visemb_features = self.visemb(features_share)
        
        if self.training:
            _, proposal_rel_losses = self.rel(visemb_features, targets_det, targets_seg, targets_rel)
            losses = {}
            losses.update(proposal_det_losses)
            losses.update(proposal_seg_losses)
            losses.update(proposal_rel_losses)
            return losses
        else:
            proposals_rel, _ = self.rel(visemb_features, proposals_det, proposals_seg)
            result = [{'det': item_det, 'seg': item_seg, 'rel': item_rel} \
                        for item_det, item_seg, item_rel in zip(proposals_det, proposals_seg, proposals_rel)]

            return result

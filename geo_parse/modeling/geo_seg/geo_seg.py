import torch
from torch import nn

from .inference import make_seg_postprocessor
from .loss import make_seg_loss_evaluator


class SEGHead(torch.nn.Module):
    
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SEGHead, self).__init__()
        
        num_classes = cfg.MODEL.SEG.NUM_CLASSES-1
        emb_dim = cfg.MODEL.SEG.EMB_DIMS

        binary_seg_tower = []
        embedding_tower = []

        for index in range(cfg.MODEL.SEG.NUM_CONVS):

            if index==0:
                in_channels_new = in_channels + 2
            else:
                in_channels_new = in_channels

            binary_seg_tower.append(
                nn.Conv2d(
                    in_channels_new,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                ))
            binary_seg_tower.append(nn.GroupNorm(32, in_channels))
            binary_seg_tower.append(nn.ReLU())

            embedding_tower.append(
                nn.Conv2d(
                    in_channels_new,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                ))
            embedding_tower.append(nn.GroupNorm(32, in_channels))
            embedding_tower.append(nn.ReLU())


        self.add_module('binary_seg_tower', nn.Sequential(*binary_seg_tower))
        self.add_module('embedding_tower', nn.Sequential(*embedding_tower))

        self.binary_seg_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1)
            
        self.embedding_pred = nn.Conv2d(
            in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):

        binary_seg = self.binary_seg_logits(self.binary_seg_tower(x))
        embedding = self.embedding_pred(self.embedding_tower(x))

        return binary_seg, embedding

class SEGModule(torch.nn.Module):
    """
        Module for SEG computation. Takes feature maps from the backbone and
        SEG outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(SEGModule, self).__init__()

        self.head = SEGHead(cfg, in_channels)
        self.loss_evaluator = make_seg_loss_evaluator(cfg)
        self.ggeo_selector = make_seg_postprocessor(cfg)
        self.cfg = cfg

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Tensor): features of last level computed from the images that are
                                used for computing the predictions. 
            targets (list[GeoList]): ground-truth masks present in the image (optional)

        Returns:
            ggeos (list[GeoList]): the predicted mask from the SEGModule, one GeoList per
                                    image.
            losses (dict[Tensor]): the losses for the model during training. During
                                    testing, it is an empty dict.
        """

        binary_seg, embedding = self.head(features)
        
        if self.training:
            return self._forward_train(
                binary_seg, embedding, targets
            )
        else:
            return self._forward_test(
                binary_seg, embedding, images.image_sizes, self.cfg.MODEL.SEG.FPN_STRIDES
            )

    def _forward_train(self,  binary_seg, embedding, targets):

        binary_seg_loss, var_loss, dist_loss, reg_loss = self.loss_evaluator(
            binary_seg, embedding, targets
        )
        losses = {
            "loss_binary_seg": binary_seg_loss * self.cfg.MODEL.SEG.LOSS_RATIO_BS,
            "loss_var": var_loss * self.cfg.MODEL.SEG.LOSS_RATIO_VAR,
            "loss_dist": dist_loss * self.cfg.MODEL.SEG.LOSS_RATIO_DIST,
            "loss_mean_reg": reg_loss * self.cfg.MODEL.SEG.LOSS_RATIO_REG
        }
        return None, losses

    def _forward_test(self, binary_seg, embedding, image_sizes, fpn_stride):

        ggeos = self.ggeo_selector(
            binary_seg, embedding, image_sizes, fpn_stride
        )
        return ggeos, {}


def build_seg(cfg, in_channels):

    return SEGModule(cfg, in_channels)

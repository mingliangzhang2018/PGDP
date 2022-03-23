import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class SegLossComputation(object):
    """
    This class computes the geo segmentation losses.
    """

    def __init__(self, cfg):

        self.discriminative_loss_func = self.discriminative_loss
        self.seg_loss_func=[]
        self.class_num = cfg.MODEL.SEG.NUM_CLASSES-1
        self.embed_dim = cfg.MODEL.SEG.EMB_DIMS
        self.delta_v = cfg.MODEL.SEG.LOSS_DELTA_V
        self.delta_d = cfg.MODEL.SEG.LOSS_DELTA_D
        for i in range(self.class_num):
            weight_ratio = torch.tensor([cfg.MODEL.SEG.CLASS_WEIGHTS[i]])
            self.seg_loss_func.append(nn.BCEWithLogitsLoss(pos_weight=weight_ratio))

    def __call__(self, binary_seg, embedding, targets):
        """
        Arguments:
            binary_seg: N*3*H*W [Tensor]
            embedding: N*EMB_DIMS*H*W [Tensor]
            targets (list[GeoList])
        Returns:
            binary_seg_loss: [Tensor]
            var_loss: [Tensor]
            dist_loss: [Tensor]
            reg_loss: [Tensor]
        """

        reshape_size = binary_seg.shape[2:]

        ########################binary_seg################################
        binary_seg_loss = binary_seg.new([0]).zero_().squeeze()
        for i in range(self.class_num):
            seg_loss_func_cur = self.seg_loss_func[i].cuda()
            input = binary_seg[:,i,:,:]
            target = []
            for ggeo in targets:
                target.append(ggeo.get_binary_seg_target(
                                        class_index = i+1,
                                        reshape_size = reshape_size)
                            )
            target = torch.from_numpy(np.stack(target,axis=0)).float().cuda()
            binary_seg_loss = binary_seg_loss + seg_loss_func_cur(input, target)

        ########################embedding_seg################################
        seg_gt = []
        for ggeo in targets:
            # get instances of line and circle
            # instance of point obtained by connected component analysis from binary_seg
            seg_gt.append(ggeo.get_inst_seg_target(
                            class_index_list = [2,3],  
                            reshape_size = reshape_size)
                        )     
        var_loss, dist_loss, reg_loss = self.discriminative_loss(embedding, seg_gt)
            
        return binary_seg_loss, var_loss, dist_loss, reg_loss

    def discriminative_loss(self, embedding, seg_gt):
        
        batch_size = embedding.shape[0]
        assert batch_size == len(seg_gt), 'embedding batch size is not equal to seg_gt batch size'

        var_loss = embedding.new([0]).zero_().squeeze()
        dist_loss = embedding.new([0]).zero_().squeeze()
        reg_loss = embedding.new([0]).zero_().squeeze()

        for b in range(batch_size):
            embedding_b = embedding[b] # [embed_dim, H, W]
            seg_gt_b = seg_gt[b]
            num_lanes = len(seg_gt_b)
            if num_lanes==0:
                continue
            centroid_mean = []

            for seg_mask_i in seg_gt_b:
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]
                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim,1) \
                                                , dim=0) - self.delta_v)**2 ) / num_lanes
            
            centroid_mean = torch.stack(centroid_mean)  # [n_lane, embed_dim]

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape [num_lanes, num_lanes]
                # diagonal elements are 0, now mask above delta_d
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=embedding.device) * self.delta_d  
                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size

        return var_loss, dist_loss, reg_loss


def make_seg_loss_evaluator(cfg):

    loss_evaluator = SegLossComputation(cfg)
    return loss_evaluator
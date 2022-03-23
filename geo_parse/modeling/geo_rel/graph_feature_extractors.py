import torch
from torch import nn
import torch.nn.functional as F
from geo_parse.modeling.poolers import Pooler
import math

class BOX2MLPFeatureExtractor(nn.Module):

    def __init__(self, cfg, in_channels, representation_size):
        super(BOX2MLPFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.REL.ROIALIGN_SIZE
        self.pooler = Pooler(output_size=resolution,
                            scales=[1.0/cfg.MODEL.VISEMB.FPN_STRIDES],
                            sampling_ratio=0)
        input_size = in_channels * resolution[0] * resolution[1]

        self.box_vis = nn.ModuleList([nn.Linear(input_size, representation_size), 
                                    nn.ReLU(),
                                    nn.Linear(representation_size, representation_size)])
        self.box_vis = nn.Sequential(*self.box_vis)

        self.USE_LOC = cfg.MODEL.REL.FEAT_WITH_LOC

        if self.USE_LOC:
            # embed box location feature [x1, y1, x2, y2]
            self.dist_meas = cfg.MODEL.VISEMB.FPN_STRIDES*(int(cfg.INPUT.MAX_SIZE_TRAIN/cfg.MODEL.SEG.FPN_STRIDES)+50)
            self.box_loc = nn.ModuleList([nn.Linear(4, representation_size), 
                                            nn.ReLU(),
                                            nn.Linear(representation_size, representation_size)])
            self.box_loc = nn.Sequential(*self.box_loc)

    def forward(self, x, proposal):
        """
        Arguments:
            x (Tensor): feature map
            proposal (BoxList): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor): feature of each bbox 
        """ 
        feat_vis = self.pooler([x.unsqueeze(0)], [proposal])
        feat_vis = feat_vis.view(feat_vis.size(0), -1)
        feat_vis = self.box_vis(feat_vis)

        if self.USE_LOC:
            feat_loc = self.box_loc(proposal.bbox/self.dist_meas)
            feat_all = feat_vis + feat_loc
        else:
            feat_all = feat_vis

        return feat_all


class GEOFeatureExtractor(nn.Module):
    
    def __init__(self, cfg, in_channels, representation_size):
        super(GEOFeatureExtractor, self).__init__()

        self.point_vis = nn.ModuleList([nn.Linear(in_channels, representation_size), 
                                            nn.ReLU(),
                                            nn.Linear(representation_size, representation_size)])
        self.point_vis = nn.Sequential(*self.point_vis)   
        self.line_vis = nn.ModuleList([nn.Linear(in_channels, representation_size), 
                                            nn.ReLU(),
                                            nn.Linear(representation_size, representation_size)])
        self.line_vis = nn.Sequential(*self.line_vis)  
        self.circle_vis = nn.ModuleList([nn.Linear(in_channels, representation_size), 
                                                nn.ReLU(),
                                                nn.Linear(representation_size, representation_size)])
        self.circle_vis = nn.Sequential(*self.circle_vis)  


        self.USE_LOC = cfg.MODEL.REL.FEAT_WITH_LOC

        if self.USE_LOC:
            # embed parsing position feature of geometric primitive 
            # point [x, y], line [x1, y1, x2, y2], circle [x, y, r]
            self.dist_meas = int(cfg.INPUT.MAX_SIZE_TRAIN/cfg.MODEL.SEG.FPN_STRIDES)+50

            self.point_loc = nn.ModuleList([nn.Linear(2, representation_size), 
                                            nn.ReLU(),
                                            nn.Linear(representation_size, representation_size)])
            self.point_loc = nn.Sequential(*self.point_loc)

            self.line_loc = nn.ModuleList([nn.Linear(4, representation_size), 
                                            nn.ReLU(),
                                            nn.Linear(representation_size, representation_size)])
            self.line_loc = nn.Sequential(*self.line_loc)  

            self.circle_loc = nn.ModuleList([nn.Linear(3, representation_size), 
                                            nn.ReLU(),
                                            nn.Linear(representation_size, representation_size)])
            self.circle_loc = nn.Sequential(*self.circle_loc)
          
    def forward(self, x, proposal):
        """
        Arguments:
            x (Tensor C*H*W): feature maps for proposal
            proposal.masks: (ndarray H*W*num): mask maps for all node
            proposal (GeoList): geometric mask boxes used to perform
        Returns:
            result (Tensor): feature of each node
        """
        locs = proposal.get_field('locs')
        labels = proposal.get_field('labels')
        masks = proposal.masks>0
        feat_list = []
        feat_map=x[:,:masks.shape[0],:masks.shape[1]].permute(1,2,0)
        device = x.device

        for index in range(masks.shape[-1]):
            if masks[:,:,index].sum()==0:  # case: masks of point is blanks due to resize operation
                x1 = math.floor(locs[index][0][1])
                y1 = math.floor(locs[index][0][0])
                x2 = math.ceil(locs[index][0][1])
                y2 = math.ceil(locs[index][0][0])
                masks[x1,y1,index]=masks[x1,y2,index]=masks[x2,y1,index]=masks[x2,y2,index]=True

            feat_vis = feat_map[masks[:,:,index],:]

            if labels[index]==1: feat_vis = self.point_vis(feat_vis)
            if labels[index]==2: feat_vis = self.line_vis(feat_vis)
            if labels[index]==3: feat_vis = self.circle_vis(feat_vis)

            if self.USE_LOC:
                if labels[index]==1: 
                    feat_loc = torch.tensor(locs[index][0], dtype=torch.float32).to(device)
                    feat_loc = self.point_loc(feat_loc/self.dist_meas)
                if labels[index]==2:
                    feat_loc = torch.tensor(locs[index][0]+locs[index][1], dtype=torch.float32).to(device)
                    feat_loc = self.line_loc(feat_loc/self.dist_meas)
                if labels[index]==3:
                    feat_loc = torch.tensor(locs[index][0]+[locs[index][1]], dtype=torch.float32).to(device)
                    feat_loc = self.circle_loc(feat_loc/self.dist_meas)
                feat_list.append(feat_vis.mean(0) + feat_loc)
            else:
                feat_list.append(feat_vis.mean(0))
                
        feat_all = torch.stack(feat_list, dim=0)

        return feat_all

    

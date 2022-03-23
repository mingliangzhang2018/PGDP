import torch
from torch import nn
from .inference import make_rel_postprocessor
from geo_parse.structures.rel_map import RelList
from geo_parse.structures.bounding_box import BoxList
from geo_parse.data.datasets.geo import GEODataset
import dgl
from .GAT import GAT
from .graph_feature_extractors import *

CLASSES_SEM = GEODataset.CLASSES_SEM

class RELModule(torch.nn.Module):
    """
        Module for REL computation. Takes feature maps from the backbone and
        REL outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg):
        super(RELModule, self).__init__()

        self.GAT = GAT(num_layers=cfg.MODEL.REL.NUM_LAYERS,
                        node_dim=cfg.MODEL.REL.NODE_DIMS,
                        edge_dim=cfg.MODEL.REL.EDGE_DIMS,
                        node_classes= cfg.MODEL.REL.NODE_NUM_CLASSES,
                        edge_classes=1
                    )
        self.rel_selector = make_rel_postprocessor(cfg)
        if cfg.MODEL.REL.FEAT_WITH_SEM:
            self.sem_embedding = nn.Embedding(cfg.MODEL.REL.SEM_NUM_CLASSES, cfg.MODEL.REL.NODE_DIMS)
        self.box_feature_extractor = BOX2MLPFeatureExtractor(cfg, cfg.MODEL.VISEMB.EMB_DIMS, cfg.MODEL.REL.NODE_DIMS)
        self.geo_feature_extractor = GEOFeatureExtractor(cfg, cfg.MODEL.VISEMB.EMB_DIMS, cfg.MODEL.REL.NODE_DIMS)
        self.edge_cls_loss_func = nn.BCEWithLogitsLoss()
        self.node_cls_loss_func = nn.CrossEntropyLoss()
        self.cfg = cfg

    def forward(self, visemb_features, targets_det, targets_seg, targets_rel=None):
        """
        Arguments:
            visemb_features (Tensor): visual embedding features computed from the images that are
                                        used for computing the predictions. (last feature level)
            targets_det (list[BoxList]): ground-truth boxes present in the image during training,
                                         but proposal boxes during testing
            targets_seg (list[GeoList]): ground-truth masks present in the image, but proposal 
                                        masks during testing
            targets_rel (list[RelList]): ground-truth relation map present in the image (training),
                                         or None (test)
        Returns:
            targets_rel (list[RelList]): the predicted relation map from the RELModule, one RelList 
                                        per image.
            losses (dict[Tensor]): the losses for the model during training. During testing, 
                                   it is an empty dict.
        """
        if not self.training:
            targets_rel = []
            for target_det, target_seg in zip(targets_det, targets_seg):
                target_rel = self.init_target_rel(target_det, target_seg)
                targets_rel.append(target_rel)

        g_list = []
        for visemb_feature, target_det, target_seg, target_rel in \
                                    zip(visemb_features, targets_det, targets_seg, targets_rel):
            g = self.construct_graph(visemb_feature, target_det, target_seg, target_rel)
            g_list.append(g)

        g_all = dgl.batch(g_list)
        if g_all.num_nodes()>0:
            g_all = self.GAT(g_all)

        if self.training:
            loss_edge =  self.edge_cls_loss_func(g_all.edata['confidence'].squeeze(dim=-1), g_all.edata['labels'])
            # mask for text node 
            mask = g_all.ndata['mask']
            loss_node = self.node_cls_loss_func(g_all.ndata['confidence'][mask], g_all.ndata['labels'][mask])
            losses = {
                "loss_edge": loss_edge * self.cfg.MODEL.REL.LOSS_RATIO_EDGE,
                "loss_node": loss_node * self.cfg.MODEL.REL.LOSS_RATIO_NODE,
            }
            return None, losses
        else:
            g_list = dgl.unbatch(g_all)
            if g_all.num_nodes()>0:
                self.rel_selector(g_list, targets_det, targets_rel)
            return targets_rel, None
        
    def init_target_rel(self, target_det, target_seg):

        ids = target_det.get_field('ids') + target_seg.get_field('ids')
        labels_all = target_det.get_field('labels').tolist() + target_seg.get_field('labels')
        labels_text = target_det.get_field('labels_text') + [-1]*len(target_seg.get_field('labels'))
        target_rel = RelList(ids, labels_all, labels_text)

        return target_rel

    def construct_graph(self, visemb_feature, target_det, target_seg, target_rel):
        '''
            construct dgl graph 
                ndata['feats'], ndata['labels'], ndata['confidence'], ndata['mask']
                edata['feats'], edata['labels'], edata['confidence']
            node features of non-geometric primitives: ROIAlign + SEM_EMBEDDING
            node features of geometric primitives: MEAN + SEM_EMBEDDING
            edge features: union-boxs for head and head_len, zeros for others 
        '''
        device = visemb_feature.device
        edge = target_rel.get_edge()
        g = dgl.graph(edge).to(device)
        target_rel.adjust_structure(g)
        # node and edge label
        g.ndata['labels'] = torch.LongTensor(target_rel.get_node_label()).to(device)
        g.edata['labels'] = torch.FloatTensor(target_rel.get_edge_label()).to(device)
        g.ndata['mask'] = torch.tensor([True if item!=-1 else False for item in target_rel.labels_text]).to(device)
        # node feat
        x_geo = self.geo_feature_extractor(visemb_feature, target_seg)
        if len(target_det)>0:
            x_sym = self.box_feature_extractor(visemb_feature, target_det)
            node_feats = torch.cat([x_sym, x_geo], dim=0)[:g.num_nodes()]
        else:
            node_feats = x_geo
        g.ndata['feats'] = node_feats[:g.num_nodes()]
        if self.cfg.MODEL.REL.FEAT_WITH_SEM:
            sem_embedding = self.sem_embedding(torch.LongTensor(target_rel.labels_sem).to(device))
            g.ndata['feats'] += sem_embedding
        # edge feat
        edge_feat_all = torch.zeros([g.num_edges(), self.cfg.MODEL.REL.EDGE_DIMS], dtype=torch.float).to(device)
        edge_boxes = []
        edge_index_list = []
        for index, (item1, item2) in enumerate(zip(edge[0], edge[1])):
            item1_id = target_rel.ids[item1]
            item2_id = target_rel.ids[item2]
            # recent only for primitives of head or head_len
            if item1_id[0]=='s' and item2_id[0]=='s': 
                x1 = min(target_det.bbox[item1][0], target_det.bbox[item2][0])
                y1 = min(target_det.bbox[item1][1], target_det.bbox[item2][1])
                x2 = max(target_det.bbox[item1][2], target_det.bbox[item2][2])
                y2 = max(target_det.bbox[item1][3], target_det.bbox[item2][3])
                edge_boxes.append([x1, y1, x2, y2])
                edge_index_list.append(index)
        if len(edge_boxes)>0:
            edge_boxes = torch.tensor(edge_boxes).to(device)
            edge_proposal = BoxList(edge_boxes, target_det.size, mode="xyxy")
            edge_feat_head = self.box_feature_extractor(visemb_feature,edge_proposal)
            edge_feat_all[edge_index_list,:]=edge_feat_head
        g.edata['feats'] = edge_feat_all
        
        return g

def build_rel(cfg):

    return RELModule(cfg)



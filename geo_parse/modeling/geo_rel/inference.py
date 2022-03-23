import numpy as np
import torch
import torch.nn.functional as F

class RelPostProcessor(torch.nn.Module):
    """
        Performs post-processing on the outputs of the relation map.
        This is only used in the testing.
    """
    def __init__(self, cfg):

        super(RelPostProcessor, self).__init__()
        self.cfg = cfg
        self.edge_thresh = cfg.MODEL.REL.EDGE_TH 
        
    def forward(self, g_list, preds_det, preds_rel):

        for g, pred_det, pred_rel in zip(g_list, preds_det, preds_rel):

            mask = g.ndata['mask']
            if mask.sum()>0:
                _, indices = torch.max(g.ndata['confidence'][mask], dim=1)
                pred_rel.construct_text_label_pred(indices.cpu(), mask.cpu())
            pred_det.extra_fields['labels_text'] = pred_rel.labels_text[:len(pred_det)]
            # predict the edge probability matrix
            conf = g.edata['confidence'].sigmoid().squeeze(-1)
            pred_rel.construct_edge_label_pred(conf.cpu(), self.edge_thresh)

def make_rel_postprocessor(cfg):

    geo_selector = RelPostProcessor(cfg)

    return geo_selector



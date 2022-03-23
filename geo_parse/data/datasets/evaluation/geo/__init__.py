import logging
from .geo_eval import do_geo_evaluation

def geo_evaluation(dataset, predictions, output_folder, iou_thresh_det, iou_thresh_seg, cfg):
    logger = logging.getLogger("geo_parse.inference")
    logger.info("performing geo evaluation, ignored iou_types.")
    return do_geo_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        iou_thresh_det=iou_thresh_det, 
        iou_thresh_seg=iou_thresh_seg,
        cfg=cfg,
        logger=logger,
    )

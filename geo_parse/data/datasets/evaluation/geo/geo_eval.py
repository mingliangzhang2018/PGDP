# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from geo_parse.structures.boxlist_ops import boxlist_iou
import shutil
import cv2
import torch
from .utils import *
from .generate_relation_rule import get_relation_rule
from .generate_relation_GAT import get_relation_GAT
from .generate_logic_form import get_logic_form
import json

def get_per_stat(Matched, numGtCare, numPredCare):

    precision = 0 if numPredCare==0 else float(len(Matched)) / numPredCare
    recall = 0 if numGtCare==0 else float(len(Matched)) / numGtCare
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, hmean

def get_per_class_stat(class_num, labels_gt, labels_pred, Matched, numGtCare, numPredCare, \
                        matchedSum, numGlobalCareGt, numGlobalCarePred, task_name):
    for class_index in range(class_num):
        if class_index==0:
            matchedSum[class_index]+=len(Matched)
            numGlobalCareGt[class_index]+=numGtCare
            numGlobalCarePred[class_index]+=numPredCare
        else:
            if task_name=='detection_text':
                label = class_index-1
            else:
                label = class_index
            matchedSum[class_index]+=np.sum(np.array(Matched)==label)
            numGlobalCareGt[class_index]+=np.sum(np.array(labels_gt)==label)
            numGlobalCarePred[class_index]+=np.sum(np.array(labels_pred)==label)

def get_all_class_stat(class_num, matchedSum, numGlobalCareGt, numGlobalCarePred, \
                        methodPrecision, methodRecall, methodHmean, \
                        logger, output_txt_file, class_name, task_name):

    for class_index in range(class_num):
        
        methodRecall[class_index] = 0 if numGlobalCareGt[class_index] == 0 else \
                                        float(matchedSum[class_index]) / numGlobalCareGt[class_index]
        methodPrecision[class_index] = 0 if numGlobalCarePred[class_index] == 0 else \
                                        float(matchedSum[class_index]) / numGlobalCarePred[class_index]
        methodHmean[class_index] = 0 if methodRecall[class_index]+methodPrecision[class_index]==0 \
                else 2 * methodRecall[class_index] * methodPrecision[class_index] / (methodRecall[class_index] + methodPrecision[class_index])

        if class_index==0:
            logger.info("-------------- "+ task_name +" --------------")
            logger.info("{:<16}\tPrecision: {:.4f}\tRecall: {:.4f}\tH-mean: {:.4f}".format( \
                            'All_Class', methodPrecision[class_index], methodRecall[class_index], methodHmean[class_index]))
            output_txt_file.write("\n-------------- "+ task_name +" --------------\n")
            output_txt_file.write("{:<16}\tPrecision: {:.4f}\tRecall: {:.4f}\tH-mean: {:.4f}\n".format( \
                        'All_Class', methodPrecision[class_index], methodRecall[class_index], methodHmean[class_index]))
        else:
            logger.info("{:<16}\tPrecision: {:.4f}\tRecall: {:.4f}\tH-mean: {:.4f}".format( \
                        class_name[class_index], methodPrecision[class_index], methodRecall[class_index], methodHmean[class_index]))
            output_txt_file.write("{:<16}\tPrecision: {:.4f}\tRecall: {:.4f}\tH-mean: {:.4f}\n".format( \
                        class_name[class_index], methodPrecision[class_index], methodRecall[class_index], methodHmean[class_index]))

def get_per_det_F1(gt_boxlist, prediction, id_gt2pred_dict, IOU_CONSTRAINT = 0.6):

    numGtCare = len(gt_boxlist)
    numPredCare = len(prediction) 
    detMatched = []

    numGtCare_text = np.sum(np.array(gt_boxlist.get_field('labels_text'))!=-1)
    numPredCare_text = np.sum(np.array(prediction.get_field('labels_text'))!=-1)
    detMatched_text = []

    if numGtCare > 0 and numPredCare > 0:
        # calculate IoU and precision matrixs all non-geometric primitives
        iouMat = np.empty([numGtCare, numPredCare])
        gtRectMat = np.zeros(numGtCare, np.int8)
        predRectMat = np.zeros(numPredCare, np.int8)
        iouMat = boxlist_iou(gt_boxlist, prediction).numpy()
        labels_gt = gt_boxlist.get_field('labels')
        labels_pred = prediction.get_field('labels')
        ids_gt = gt_boxlist.get_field('ids')
        ids_pred = prediction.get_field('ids')
        # calculate IoU and precision matrixs of text
        gtRectMat_text = np.zeros(numGtCare, np.int8)
        predRectMat_text = np.zeros(numPredCare, np.int8)
        labels_gt_text = gt_boxlist.get_field('labels_text')
        labels_pred_text = prediction.get_field('labels_text')

    for gtNum in range(numGtCare):

        for predNum in range(numPredCare):
            if gtRectMat[gtNum] == 0 and predRectMat[predNum] == 0 and  \
                labels_gt[gtNum]==labels_pred[predNum] and iouMat[gtNum, predNum] > IOU_CONSTRAINT:
                gtRectMat[gtNum] = 1
                predRectMat[predNum] = 1
                detMatched.append(labels_gt[gtNum])
                # box location and fine-grained class are all matched
                if labels_gt_text[gtNum]==labels_pred_text[predNum]:
                    id_gt2pred_dict[ids_gt[gtNum]]=ids_pred[predNum]
                break

        for predNum in range(numPredCare):
            if gtRectMat_text[gtNum] == 0 and predRectMat_text[predNum] == 0 and  \
                labels_gt_text[gtNum]==labels_pred_text[predNum] and labels_gt_text[gtNum]!=-1 and \
                    iouMat[gtNum, predNum] > IOU_CONSTRAINT:
                gtRectMat_text[gtNum] = 1
                predRectMat_text[predNum] = 1
                detMatched_text.append(labels_gt_text[gtNum])
                break
    
    return detMatched, numGtCare, numPredCare, detMatched_text, numGtCare_text, numPredCare_text

def get_per_seg_F1(gt_geolist, prediction, id_gt2pred_dict, IOU_CONSTRAINT = 0.75):
    """
    return 
        segMatched: label list of matched pair 
        match method:
                    |     GT     | Pred
            -----------------------------------
            point   | point_loc  | mask
            line    | mask(tk=1) | mask
            circle  | mask(tk=1) | mask
    """
    def get_round_loc(loc):
        loc_copy = []
        for item in loc:
            if isinstance(item, float):
                item = round(item)
            if isinstance(item, list):
                item = tuple([round(v) for v in item])
            loc_copy.append(item)
        return loc_copy 

    numGtCare = len(gt_geolist)
    numPredCare = len(prediction) 
    segMatched = []

    if numGtCare > 0 and numPredCare > 0:
        gtRectMat = np.zeros(numGtCare, np.int8)
        predRectMat = np.zeros(numPredCare, np.int8)

        labels_gt = gt_geolist.get_field('labels')
        labels_pred = prediction.get_field('labels')
        ids_gt = gt_geolist.get_field('ids')
        ids_pred = prediction.get_field('ids')

        gt_img_mask = []
        gt_area = []
        pred_img_mask = []

        for gtNum in range(numGtCare):
            loc = get_round_loc(gt_geolist.extra_fields['locs'][gtNum])
            gt_img = np.zeros((gt_geolist.size[1], gt_geolist.size[0]), dtype=np.uint8)
            if labels_gt[gtNum]==1: # point
                gt_img[loc[0][1],loc[0][0]]=255
            elif labels_gt[gtNum]==2: # line
                cv2.line(gt_img, pt1=loc[0], pt2=loc[1], color=255, thickness=1)
            elif labels_gt[gtNum]==3: # circle
                cv2.circle(gt_img, center=loc[0], radius=loc[1], color=255, thickness=1)
                if loc[2]!='1111': # handle the case of nonholonomic circle
                    draw_pixel_loc = np.where(gt_img>0)
                    for x, y in zip(draw_pixel_loc[1], draw_pixel_loc[0]):
                        delta_x = x-loc[0][0] # + - - +
                        delta_y = y-loc[0][1] # - - + +
                        if delta_x>=0 and delta_y<=0 and loc[2][0]=='0': gt_img[y,x] = 0
                        if delta_x<=0 and delta_y<=0 and loc[2][1]=='0': gt_img[y,x] = 0
                        if delta_x<=0 and delta_y>=0 and loc[2][2]=='0': gt_img[y,x] = 0
                        if delta_x>=0 and delta_y>=0 and loc[2][3]=='0': gt_img[y,x] = 0

            mask = gt_img>0
            gt_img_mask.append(mask)
            gt_area.append(np.sum(mask))

        for predNum in range(numPredCare):
            mask = prediction.masks[:,:,predNum]>0 
            pred_img_mask.append(mask)
    
    for gtNum in range(numGtCare):
        for predNum in range(numPredCare):
            if gtRectMat[gtNum] == 0 and predRectMat[predNum] == 0 and  \
                labels_gt[gtNum]==labels_pred[predNum]:
                union_mask = gt_img_mask[gtNum]&pred_img_mask[predNum]
                if np.sum(union_mask)/gt_area[gtNum]>IOU_CONSTRAINT:
                    gtRectMat[gtNum] = 1
                    predRectMat[predNum] = 1
                    segMatched.append(labels_gt[gtNum])
                    id_gt2pred_dict[ids_gt[gtNum]]=ids_pred[predNum]
                    break

    return segMatched, numGtCare, numPredCare

def get_per_rel_F1(gt_relation, prediction, id_gt2pred_dict, CLASSES_SEM, CLASSES_REL):
    '''
        the performance of edge (two-node pair)
    '''
    def judge_rel_label(item):
        rel_name = "geo2geo"
        name1 = CLASSES_SEM[item[0]]
        name2 = CLASSES_SEM[item[1]]
        for x, y in [(name1, name2), (name2, name1)]:  
            if (x in ['point','line', 'circle']) and (y in ['point','line', 'circle']):
                rel_name = "geo2geo"
            if (x=='text') and (y in ['point', 'line', 'circle']):
                rel_name = "text2geo"
            if (not x in ['point', 'line', 'circle','text']) and (y in ['point', 'line', 'circle']):
                rel_name = "sym2geo"
            if (x=='text') and (not y in ['point', 'line', 'circle', 'text']):
                rel_name = "text2head"
        return CLASSES_REL.index(rel_name) 

    loc_gt = np.where(gt_relation.edge_label==1)
    loc_pred = np.where(prediction.edge_label==1)

    gt_pair, pred_pair = [], []
    relMatched = []
    gt_rel_label, pred_rel_label = [], []

    for i, j in zip(loc_gt[0], loc_gt[1]):
        id_item = [gt_relation.ids[i], gt_relation.ids[j]]
        gt_pair.append(id_item)
        sem_item = [gt_relation.labels_sem[i], gt_relation.labels_sem[j]]
        gt_rel_label.append(judge_rel_label(sem_item))

    for i, j in zip(loc_pred[0], loc_pred[1]):
        id_item = [prediction.ids_t[i], prediction.ids_t[j]]
        pred_pair.append(id_item)
        sem_item = [prediction.labels_sem_t[i], prediction.labels_sem_t[j]]
        pred_rel_label.append(judge_rel_label(sem_item))

    numGtCare = len(gt_pair)
    numPredCare = len(pred_pair)

    for pair, label in zip(gt_pair, gt_rel_label):
        if pair[0] in id_gt2pred_dict and pair[1] in id_gt2pred_dict and \
            [id_gt2pred_dict[pair[0]],id_gt2pred_dict[pair[1]]] in pred_pair:
            relMatched.append(label)

    return relMatched, numGtCare, numPredCare, gt_rel_label, pred_rel_label

def do_geo_evaluation(dataset, predictions, output_folder, 
                        iou_thresh_det, iou_thresh_seg, logger, cfg):

    output_im_dir = os.path.join(output_folder, 'img_results')
    output_txt_file = open(os.path.join(output_folder, "results.txt"), "w")
    output_txt_file.write("File_Name\tPrecision_det\tRecall_det\tH-mean_det\tPrecision_det_text\tRecall_det_text\tH-mean_det_text\tPrecision_seg\tRecall_seg\tH-mean_seg\tPrecision_rel\tRecall_rel\tH-mean_rel\n")
    output_relation_path = os.path.join(output_folder, "relations_pred.json")
    output_logic_form_path = os.path.join(output_folder, "logic_forms_pred.json")
    
    if os.path.exists(output_im_dir):
        shutil.rmtree(output_im_dir)
    os.makedirs(output_im_dir)
    
    class_num_det = len(dataset.CLASSES_SYM)
    class_num_seg= len(dataset.CLASSES_GEO)
    class_num_det_text = len(dataset.CLASSES_TEXT)+1
    class_num_rel= len(dataset.CLASSES_REL)

    matchedSum_det = [0]*class_num_det
    matchedSum_seg = [0]*class_num_seg
    matchedSum_det_text = [0]*class_num_det_text
    matchedSum_rel = [0]*class_num_rel

    numGlobalCareGt_det = [0]*class_num_det
    numGlobalCareGt_seg = [0]*class_num_seg
    numGlobalCareGt_det_text = [0]*class_num_det_text
    numGlobalCareGt_rel = [0]*class_num_rel

    numGlobalCarePred_det = [0]*class_num_det
    numGlobalCarePred_seg = [0]*class_num_seg
    numGlobalCarePred_det_text = [0]*class_num_det_text
    numGlobalCarePred_rel = [0]*class_num_rel

    methodRecall_det = [0]*class_num_det
    methodRecall_seg = [0]*class_num_seg
    methodRecall_det_text = [0]*class_num_det_text
    methodRecall_rel = [0]*class_num_rel

    methodPrecision_det = [0]*class_num_det
    methodPrecision_seg = [0]*class_num_seg
    methodPrecision_det_text = [0]*class_num_det_text
    methodPrecision_rel = [0]*class_num_rel

    methodHmean_det = [0]*class_num_det
    methodHmean_seg = [0]*class_num_seg
    methodHmean_det_text = [0]*class_num_det_text
    methodHmean_rel = [0]*class_num_rel

    error_num_det = 0
    error_num_det_text = 0 
    error_num_seg = 0 
    error_num_rel = 0
    error_num_all = 0
    relations = dict()
    logic_forms = dict()

    for image_id, prediction in enumerate(predictions):

        o=False
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        img_id = dataset.ids[image_id]
        annot_each = dataset.contents[img_id]
        file_name = annot_each['file_name']

        gt_boxlist, gt_geolist, gt_relation = dataset.get_groundtruth(image_id)

        #  dilate the mask for accurate evaluation of geometric primitive segmentation 
        prediction['seg'].masks = cv2.dilate(prediction['seg'].masks, 
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) 
        prediction['det'] = prediction['det'].resize((image_width, image_height))
        prediction['det'] = prediction['det'].to(torch.device("cpu"))
        prediction['seg'] = prediction['seg'].resize((image_width, image_height))
  
        id2name_dict = None
        id_gt2pred_dict = {}
        # relation_each, elem_dict = get_relation_rule(prediction, file_name, gt_boxlist, dataset.CLASSES_SYM)
        relation_each, elem_dict = get_relation_GAT(prediction, file_name, gt_boxlist, cfg)
        logic_form_each, id2name_dict = get_logic_form(relation_each, elem_dict, img_id)

        img = cv2.imread(os.path.join(dataset.img_root, file_name))
        if not img is None:
            show_img_result(img, file_name, dataset.CLASSES_SYM, dataset.CLASSES_GEO, 
                                prediction, output_im_dir, id2name_dict)

        relations[img_id] = relation_each
        logic_forms[img_id] = logic_form_each
        # reconstruct the edge label after previous operations (reconstruct hyperedges of diagram)
        prediction['rel'].reconstruct_edge_label(relation_each['relations'])

        ##################################  detection evaluation #################################
        detMatched, numGtCare_det, numPredCare_det, detMatched_text, \
                    numGtCare_det_text, numPredCare_det_text = \
                    get_per_det_F1(gt_boxlist, prediction['det'], id_gt2pred_dict, IOU_CONSTRAINT=iou_thresh_det)
        precision_det, recall_det, hmean_det = get_per_stat(detMatched, numGtCare_det, numPredCare_det)
        get_per_class_stat(class_num_det, \
                        gt_boxlist.get_field('labels'), prediction['det'].get_field('labels'), \
                        detMatched, numGtCare_det, numPredCare_det, \
                        matchedSum_det, numGlobalCareGt_det, numGlobalCarePred_det, \
                        task_name='detection_all')
        if numGtCare_det!=0 and hmean_det!=1:
            error_num_det=error_num_det+1
            o = True
        ################################# text detection evaluation #################################
        precision_det_text, recall_det_text, hmean_det_text = get_per_stat(detMatched_text, numGtCare_det_text, numPredCare_det_text)
        get_per_class_stat(class_num_det_text, \
                        gt_boxlist.get_field('labels_text'), prediction['det'].get_field('labels_text'), \
                        detMatched_text, numGtCare_det_text, numPredCare_det_text, \
                        matchedSum_det_text, numGlobalCareGt_det_text, numGlobalCarePred_det_text, \
                        task_name='detection_text')
        if numGtCare_det_text!=0 and hmean_det_text!=1:
            error_num_det_text=error_num_det_text+1
            o = True
        ################################# segmentation evaluation #################################
        segMatched, numGtCare_seg, numPredCare_seg = get_per_seg_F1(gt_geolist, prediction['seg'], \
                                                                    id_gt2pred_dict, IOU_CONSTRAINT=iou_thresh_seg)
        precision_seg, recall_seg, hmean_seg = get_per_stat(segMatched, numGtCare_seg, numPredCare_seg)
        get_per_class_stat(class_num_seg, \
                        gt_geolist.get_field('labels'), prediction['seg'].get_field('labels'), \
                        segMatched, numGtCare_seg, numPredCare_seg, \
                        matchedSum_seg, numGlobalCareGt_seg, numGlobalCarePred_seg, \
                        task_name='segmentation')      
        if  numGtCare_seg!=0 and hmean_seg!=1:
            error_num_seg=error_num_seg+1
            o = True
        ################################## relation evaluation ###################################
        relMatched, numGtCare_rel, numPredCare_rel, gt_rel_label, pred_rel_label = \
                                    get_per_rel_F1(gt_relation, prediction['rel'], id_gt2pred_dict, \
                                                    dataset.CLASSES_SEM, dataset.CLASSES_REL)
        precision_rel, recall_rel, hmean_rel = get_per_stat(relMatched, numGtCare_rel, numPredCare_rel)
        get_per_class_stat(class_num_rel, \
                            gt_rel_label, pred_rel_label, \
                            relMatched, numGtCare_rel, numPredCare_rel, \
                            matchedSum_rel, numGlobalCareGt_rel, numGlobalCarePred_rel, \
                            task_name='relation')      
        if  numGtCare_rel!=0 and hmean_rel!=1:
            error_num_rel=error_num_rel+1
            o = True
        ############################################################################################

        if o: error_num_all+=1

        output_txt_file.write("{:s}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n". \
                            format(annot_each['file_name'], \
                                    precision_det, recall_det, hmean_det, \
                                    precision_det_text, recall_det_text, hmean_det_text, \
                                    precision_seg, recall_seg, hmean_seg, \
                                    precision_rel, recall_rel, hmean_rel)) 

    get_all_class_stat(class_num_det, matchedSum_det, numGlobalCareGt_det, numGlobalCarePred_det, \
                        methodPrecision_det, methodRecall_det, methodHmean_det, \
                        logger, output_txt_file, dataset.CLASSES_SYM, 'detection_all')
    
    get_all_class_stat(class_num_det_text, matchedSum_det_text, numGlobalCareGt_det_text, numGlobalCarePred_det_text, \
                        methodPrecision_det_text, methodRecall_det_text, methodHmean_det_text, \
                        logger, output_txt_file, ['All']+dataset.CLASSES_TEXT, 'detection_text')

    get_all_class_stat(class_num_seg, matchedSum_seg, numGlobalCareGt_seg, numGlobalCarePred_seg, \
                        methodPrecision_seg, methodRecall_seg, methodHmean_seg, \
                        logger, output_txt_file, dataset.CLASSES_GEO, 'segmentation')
    
    get_all_class_stat(class_num_rel, matchedSum_rel, numGlobalCareGt_rel, numGlobalCarePred_rel, \
                        methodPrecision_rel, methodRecall_rel, methodHmean_rel, \
                        logger, output_txt_file, dataset.CLASSES_REL, 'relation')

    logger.info("error det number: "+str(error_num_det))
    logger.info("error det text number: "+str(error_num_det_text))
    logger.info("error seg number: "+str(error_num_seg))
    logger.info("error rel number: "+str(error_num_rel))
    logger.info("error all number: "+str(error_num_all))

    output_txt_file.write("\nerror det number: "+str(error_num_det)+'\n')
    output_txt_file.write("error det text number: "+str(error_num_det_text)+'\n')
    output_txt_file.write("error seg number: "+str(error_num_seg)+'\n')
    output_txt_file.write("error rel number: "+str(error_num_rel)+'\n')
    output_txt_file.write("error all number: "+str(error_num_all)+'\n')

    output_txt_file.close()

    with open(output_relation_path, 'w') as f:
        json.dump(relations, f, indent=4)
    with open(output_logic_form_path, 'w') as f:
        json.dump(logic_forms, f, indent=2)

    

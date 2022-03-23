import numpy as np
from geo_parse.structures.boxlist_ops import boxlist_iou
from .utils import *
from geo_parse.data.datasets.geo import GEODataset

CLASSES_SYM = GEODataset.CLASSES_SYM
CLASSES_GEO = GEODataset.CLASSES_GEO
CLASSES_TEXT = GEODataset.CLASSES_TEXT
CLASSES_SEM = GEODataset.CLASSES_SEM


def get_geos_pred_gnn(geos, pred_seg, elem_dict):
    points, circles, lines = [], [], []
    for class_index, loc, id in zip(pred_seg.get_field("labels"), pred_seg.get_field("locs"), \
                                    pred_seg.get_field("ids")):
        dict_each = dict()
        if class_index==1: # point
            dict_each['id'] = id
            dict_each['loc'] = loc
            points.append(dict_each)
        elif class_index==3: # circle
            dict_each['id'] = id
            dict_each['loc'] = loc
            circles.append(dict_each)
        elif class_index==2: # line
            dict_each['id'] = id
            lines.append(dict_each)
        if dict_each: elem_dict[dict_each['id']] = dict_each

    geos["points"] = points
    geos["circles"] = circles
    geos["lines"] = lines

def get_geos_pred_rule(geos, pred_seg, elem_dict, geo_mask):
    points, circles = [], []
    for index, (class_index, loc, id) in enumerate(zip(pred_seg.get_field("labels"), pred_seg.get_field("locs"), \
                                                    pred_seg.get_field("ids"))):
        dict_each = dict()
        if class_index==1:  # point
            dict_each['id'] = id
            dict_each['loc'] = loc
            points.append(dict_each)
            geo_mask['points'][dict_each['id']] = pred_seg.masks[:,:,index]
        elif class_index==3: # circle
            dict_each['id'] = id
            dict_each['loc'] = loc
            circles.append(dict_each)
            geo_mask['circles'][dict_each['id']] = pred_seg.masks[:,:,index]
        elif class_index==2: # line
            geo_mask['lines'][id] = pred_seg.masks[:,:,index]
        if dict_each: elem_dict[dict_each['id']] = dict_each

    geos["points"] = points
    geos["circles"] = circles

def get_geo2geo_relation_gnn(geos, elem_dict, rel_map, rela_geo2geo):
    point_on_line = dict()
    id_lines = [item_line['id'] for item_line in geos['lines']]
    id_circles = [item_circle['id'] for item_circle in geos['circles']]

    for id_line in id_lines:
        _, id_node_list = get_connected_node(id_line, rel_map, CLASSES_SEM, name_list=['point'])
        point_on_line[id_line]=id_node_list
        
    lines = []
    for id_line in id_lines:
        if len(point_on_line[id_line])>0:
            dict_each = dict()
            # the points on the line are sorted along the x or y axes (dimensions with large variances)
            coord_x = [elem_dict[point]['loc'][0][0] for point in point_on_line[id_line]]
            coord_y = [elem_dict[point]['loc'][0][1] for point in point_on_line[id_line]]
            if max(coord_x)-min(coord_x)>max(coord_y)-min(coord_y):
                point_on_line[id_line].sort(key=lambda x:elem_dict[x]['loc'][0][0])
            else:
                point_on_line[id_line].sort(key=lambda x:elem_dict[x]['loc'][0][1])

            for index_point, id_point in enumerate(point_on_line[id_line]):
                if index_point==0 or index_point==len(point_on_line[id_line])-1:
                    rela_geo2geo.append([id_point, id_line, 'endpoint'])
                else:
                    rela_geo2geo.append([id_point, id_line, 'online'])
            dict_each['id'] = id_line
            dict_each['loc'] = elem_dict[point_on_line[id_line][0]]['loc'] + elem_dict[point_on_line[id_line][-1]]['loc']
            lines.append(dict_each)
    geos["lines"] = lines       

    for id_circle in id_circles:
        _, id_node_list = get_connected_node(id_circle, rel_map, CLASSES_SEM, name_list=['point'])
        center_loc = elem_dict[id_circle]['loc'][0]
        radius = elem_dict[id_circle]['loc'][1]
        for id_point in id_node_list:
            point_loc = elem_dict[id_point]['loc'][0]
            dist_c2p = get_point_dist(center_loc, point_loc)
            if abs(dist_c2p-radius)<dist_c2p:
                rela_geo2geo.append([id_point, id_circle, "oncircle"])
            else:
                rela_geo2geo.append([id_point, id_circle, "center"])

def get_geo2geo_relation_rule(geos, elem_dict, rela_geo2geo, geo_mask):
    point_on_line = dict()
    for key in geo_mask['lines']:
        point_on_line[key] = []

    # the endpoints of line are determined by the point instances
    for index_line, mask_line in geo_mask['lines'].items():
        mask_loc = np.where(mask_line>0)
        x_min, x_min_index = np.min(mask_loc[1]), np.argmin(mask_loc[1])
        x_max, x_max_index = np.max(mask_loc[1]), np.argmax(mask_loc[1])
        y_min, y_min_index = np.min(mask_loc[0]), np.argmin(mask_loc[0])
        y_max, y_max_index = np.max(mask_loc[0]), np.argmax(mask_loc[0])
        if x_max-x_min>y_max-y_min:
            endpoint_loc_1 = [x_min, mask_loc[0][x_min_index]]
            endpoint_loc_2 = [x_max, mask_loc[0][x_max_index]]
        else:
            endpoint_loc_1 = [mask_loc[1][y_min_index], y_min]
            endpoint_loc_2 = [mask_loc[1][y_max_index], y_max]
        nearst_point = None
        for endpoint_loc in [endpoint_loc_1, endpoint_loc_2]:
            point_list = [(index_point, get_point_dist(elem_dict[index_point]['loc'][0], endpoint_loc))  \
                                                                for index_point in geo_mask['points']]
            point_list.sort(key=lambda x:x[1])
            if point_list[0][0]!=nearst_point: 
                nearst_point = point_list[0][0]
            else:
                nearst_point = point_list[1][0]
            if not nearst_point is None:
                point_on_line[index_line].append(nearst_point)

    # determine the point on line by the principle of mask intersection
    for index_point, mask_point in geo_mask['points'].items():
            for index_line, mask_line in geo_mask['lines'].items():
                if not index_point in point_on_line[index_line] and \
                    np.sum((mask_point>0)&(mask_line>0))>0:
                    point_on_line[index_line].append(index_point)
                
    lines = []
    for key in point_on_line:
        dict_each = dict()
        coord_x = [elem_dict[point]['loc'][0][0] for point in point_on_line[key]]
        coord_y = [elem_dict[point]['loc'][0][1] for point in point_on_line[key]]
        if max(coord_x)-min(coord_x)>max(coord_y)-min(coord_y):
            point_on_line[key].sort(key=lambda x:elem_dict[x]['loc'][0][0])
        else:
            point_on_line[key].sort(key=lambda x:elem_dict[x]['loc'][0][1])

        for index_point, point in enumerate(point_on_line[key]):
            if index_point==0 or index_point==len(point_on_line[key])-1:
                rela_geo2geo.append([point, key, 'endpoint'])
            else:
                rela_geo2geo.append([point, key, 'online'])
        dict_each['id'] = key
        dict_each['loc'] = elem_dict[point_on_line[key][0]]['loc'] + elem_dict[point_on_line[key][-1]]['loc']
        lines.append(dict_each)
    geos["lines"] = lines       

    # determine relation of point to circle by distance rules  
    for index_point, mask_point in geo_mask['points'].items():
        point_loc = elem_dict[index_point]['loc'][0]
        for index_circle, _ in geo_mask['circles'].items():
            center_loc = elem_dict[index_circle]['loc'][0]
            radius = elem_dict[index_circle]['loc'][1]
            if abs(get_point_dist(center_loc, point_loc)-radius)<THRESH_DIST:
                rela_geo2geo.append([index_point, index_circle, "oncircle"])
            elif get_point_dist(center_loc, point_loc)<THRESH_DIST:
                rela_geo2geo.append([index_point, index_circle, "center"])

def get_syms_pred(symbols, pred_det, gt_boxlist, elem_dict):
    
    # get the content of text box, using the GT in default
    if len(pred_det)>0 and len(gt_boxlist)>0:
        iouMat = boxlist_iou(pred_det, gt_boxlist).numpy()
        text_content_gt = gt_boxlist.get_field("text_contents")
        text_content_pred = [text_content_gt[index] if value>0 else None \
                        for index, value in zip(np.argmax(iouMat, axis=1), np.max(iouMat, axis=1)) ]
    else:
        text_content_pred=[None]*len(pred_det)

    for class_index, text_index, id, box, text_content in zip(pred_det.get_field("labels"), pred_det.get_field("labels_text"), \
                                                    pred_det.get_field("ids"), pred_det.bbox, text_content_pred):
        dict_each = dict()
        dict_each['id'] = id
        class_name = CLASSES_SYM[class_index]
        if class_name == 'text':
            dict_each["sym_class"] = 'text'
            dict_each["text_class"] = CLASSES_TEXT[text_index]
            dict_each["text_content"] = text_content
        else:
            dict_each["sym_class"] = class_name
            dict_each["text_class"] = "others"
            dict_each["text_content"] = None
        box_xyxy = box.tolist()
        # convert to the form of xywh 
        dict_each["bbox"] = [box_xyxy[0], box_xyxy[1], box_xyxy[2]-box_xyxy[0]+1, box_xyxy[3]-box_xyxy[1]+1]
        symbols.append(dict_each)
        elem_dict[dict_each['id']] = dict_each

def get_textpoint2point_relation(symbols, rel_map, rela_sym2geo, rela_sym2sym):

    id_texts = get_name_id('point', symbols, class_name ='text_class')

    for id_text in id_texts:
        name_node_list, id_node_list = get_connected_node(id_text, rel_map, CLASSES_SEM)
        if is_meet_cond(name_node_list, ['head']):
            rela_sym2sym.append([id_text, id_node_list])
            id_head = id_node_list[0]
            name_node_list, id_node_list = get_connected_node(id_head, rel_map, CLASSES_SEM, name_list=['point'])
            if is_meet_cond(name_node_list, ['point']):
                rela_sym2geo.append([id_head, id_node_list])
        elif is_meet_cond(name_node_list, ['point']):
            rela_sym2geo.append([id_text, id_node_list])
        
def get_textlen2point_relation(symbols, rel_map, rela_sym2geo, rela_sym2sym, elem_dict):

    id_texts = get_name_id('len', symbols, class_name='text_class')
    id_point_list = [item for item in rel_map.ids if item[0]=='p']

    for id_text in id_texts:
        name_node_list, id_node_list = get_connected_node(id_text, rel_map, CLASSES_SEM)
        if is_meet_cond(name_node_list, ['head']):
            rela_sym2sym.append([id_text, id_node_list])
            id_head = id_node_list[0]
            name_node_list, id_node_list = get_connected_node(id_head, rel_map, CLASSES_SEM, \
                                                    name_list=['point', 'line', 'circle', 'head_len'])
            if is_meet_cond(name_node_list, ['point', 'point', 'line']) or \
                            is_meet_cond(name_node_list, ['point', 'point', 'circle']):
                rela_sym2geo.append([id_head, id_node_list])
            if is_meet_cond(name_node_list, ['head_len', 'head_len']):
                rela_sym2sym.append([id_head, id_node_list])
                cand_point_ids = get_head_len2point(elem_dict, id_node_list, id_point_list)
                if cand_point_ids:
                    rela_sym2geo.append([id_node_list[0], [cand_point_ids[0]]])
                    rela_sym2geo.append([id_node_list[1], [cand_point_ids[1]]])
        elif is_meet_cond(name_node_list, ['head_len', 'head_len']):
            rela_sym2sym.append([id_text, id_node_list])
            cand_point_ids = get_head_len2point(elem_dict, id_node_list, id_point_list)
            if cand_point_ids:
                rela_sym2geo.append([id_node_list[0], [cand_point_ids[0]]])
                rela_sym2geo.append([id_node_list[1], [cand_point_ids[1]]])
        elif is_meet_cond(name_node_list, ['point', 'point', 'line']) or \
                            is_meet_cond(name_node_list, ['point', 'point', 'circle']):
            rela_sym2geo.append([id_text, id_node_list])    

def get_textline2line_relation(symbols, rel_map, rela_sym2geo):

    id_texts = get_name_id('line', symbols, class_name='text_class')
    
    for id_text in id_texts:
        name_node_list, id_node_list = get_connected_node(id_text, rel_map, CLASSES_SEM)
        if is_meet_cond(name_node_list, ['point', 'line']):
            rela_sym2geo.append([id_text, id_node_list])

def get_parallel2line_relation(symbols, rel_map, rela_sym2geo):
    
    id_syms = get_name_id('parallel', symbols, class_name='sym_class')

    for id_sym in id_syms:
        name_node_list, id_node_list = get_connected_node(id_sym, rel_map, CLASSES_SEM)
        if is_meet_cond(name_node_list, ['line']):
            rela_sym2geo.append([id_sym, id_node_list])

def get_bar2point_relation(symbols, rel_map, rela_sym2geo):
    
    id_syms = get_name_id('bar', symbols, class_name='sym_class')
    for id_sym in id_syms:
        name_node_list, id_node_list = get_connected_node(id_sym, rel_map, CLASSES_SEM)
        if is_meet_cond(name_node_list, ['point', 'point', 'line']) or \
                        is_meet_cond(name_node_list, ['point', 'point', 'circle']):
            rela_sym2geo.append([id_sym, id_node_list])    

def get_perpendicular_relation(symbols, rel_map, rela_sym2geo):

    id_syms = get_name_id('perpendicular', symbols, class_name='sym_class')

    for id_sym in id_syms:
        name_node_list, id_node_list = get_connected_node(id_sym, rel_map, CLASSES_SEM)
        if is_meet_cond(name_node_list, ['point', 'line', 'line']):
            rela_sym2geo.append([id_sym, id_node_list])    

def get_angle_relation(symbols, rel_map, rela_sym2geo, rela_sym2sym, check_name, class_name):

    ids = get_name_id(check_name, symbols, class_name=class_name)

    for id in ids:
        name_node_list, id_node_list = get_connected_node(id, rel_map, CLASSES_SEM)
        if is_meet_cond(name_node_list, ['head']):
            rela_sym2sym.append([id, id_node_list])
            id_head = id_node_list[0]
            name_node_list, id_node_list = get_connected_node(id_head, rel_map, CLASSES_SEM, name_list=['point', 'line', 'circle'])
            if is_meet_cond(name_node_list, ['point', 'line', 'line']) or \
                            is_meet_cond(name_node_list, ['point', 'point', 'circle']):
                rela_sym2geo.append([id_head, id_node_list])
        elif is_meet_cond(name_node_list, ['point', 'line', 'line']) or \
                            is_meet_cond(name_node_list, ['point', 'point', 'circle']):
            rela_sym2geo.append([id, id_node_list]) 

def get_relation_GAT(prediction, file_name, gt_boxlist, cfg):

    relation_each = dict()
    elem_dict = dict()
    geos = {}
    symbols = []
    relations = dict()
    relations['geo2geo'] = []
    relations['sym2geo'] = []
    relations['sym2sym'] = []
    relation_each['file_name'] = file_name
    relation_each['width'] = prediction['seg'].size[0]
    relation_each['height'] = prediction['seg'].size[1]

    geo_mask = dict()
    geo_mask['points'] = {}
    geo_mask['lines'] = {}
    geo_mask['circles'] = {}
    
    if cfg.MODEL.REL.GEO2GEO_RULE:
        # determine the geo2geo relationship by relation of instance masks 
        get_geos_pred_rule(geos, prediction['seg'], elem_dict, geo_mask)
        get_geo2geo_relation_rule(geos, elem_dict, relations['geo2geo'], geo_mask)
    else:
        # determine the geo2geo relationship by GNN
        get_geos_pred_gnn(geos, prediction['seg'], elem_dict)
        get_geo2geo_relation_gnn(geos, elem_dict, prediction['rel'], relations['geo2geo'])

    relation_each['geos'] = geos
    get_syms_pred(symbols, prediction['det'], gt_boxlist, elem_dict)
    relation_each['symbols'] = symbols
    get_textpoint2point_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'], relations['sym2sym'])
    get_textlen2point_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'], relations['sym2sym'], elem_dict)
    get_textline2line_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'])
    get_parallel2line_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'])
    get_bar2point_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'])
    get_perpendicular_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'])
    get_angle_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'], relations['sym2sym'], 'angle', 'sym_class')
    get_angle_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'], relations['sym2sym'], 'angle', 'text_class')
    get_angle_relation(relation_each['symbols'], prediction['rel'], relations['sym2geo'], relations['sym2sym'], 'degree', 'text_class')

    relation_each['relations'] = relations

    return relation_each, elem_dict


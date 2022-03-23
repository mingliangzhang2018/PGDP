import numpy as np
from geo_parse.structures.boxlist_ops import boxlist_iou
from .utils import *
from itertools import combinations


def get_geos_pred(geos, pred_seg, elem_dict, geo_mask):
    points, circles = [], []
    for index, (class_index, loc, id) in enumerate(zip(pred_seg.get_field("labels"), pred_seg.get_field("locs"), \
                                                        pred_seg.get_field("ids"))):
        dict_each = dict()
        if class_index==1: # point
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

def get_syms_pred(symbols, pred_det, gt_boxlist, elem_dict, CLASSES_SYM):

    # get the content of text box, using the GT in default
    if len(pred_det)>0 and len(gt_boxlist)>0:
        iouMat = boxlist_iou(pred_det, gt_boxlist).numpy()
        text_content_gt = gt_boxlist.get_field("text_contents")
        text_content_pred = [text_content_gt[index] if value>0 else None \
                        for index, value in zip(np.argmax(iouMat, axis=1), np.max(iouMat, axis=1)) ]
    else:
        text_content_pred=[None]*len(pred_det)

    for class_index, id, box, text_content in zip(pred_det.get_field("labels"), pred_det.get_field("ids"), \
                                                    pred_det.bbox, text_content_pred):
        dict_each = dict()
        dict_each['id'] = id

        class_name = CLASSES_SYM[class_index]

        if class_name.split('_')[0] == 'text':
            dict_each["sym_class"] = 'text'
            dict_each["text_class"] = class_name.split('_')[1]
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

def get_geo2geo_relation(geos, elem_dict, geo_mask, rel_map, relations):

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
                relations.append([point, key, 'endpoint'])
            else:
                relations.append([point, key, 'online'])
        dict_each['id'] = key
        dict_each['loc'] = elem_dict[point_on_line[key][0]]['loc'] + elem_dict[point_on_line[key][-1]]['loc']
        lines.append(dict_each)
    geos["lines"] = lines       

    # determine relation of point to circle by distance rules  
    for index_point, mask_point in geo_mask['points'].items():
        point_loc = elem_dict[index_point]['loc'][0]
        for index_circle, mask_circle in geo_mask['circles'].items():

            center_loc = elem_dict[index_circle]['loc'][0]
            radius = elem_dict[index_circle]['loc'][1]
            if abs(get_point_dist(center_loc, point_loc)-radius)<THRESH_DIST:
                relations.append([index_point, index_circle, "oncircle"])
            elif get_point_dist(center_loc, point_loc)<THRESH_DIST:
                relations.append([index_point, index_circle, "center"])

def get_textpoint2point_relation(points, symbols, relations):
    """
        nearest neighbor rule -- (point) to (text)
    """
    need_items  = get_name_item('point', symbols, class_name='text_class')
    vist_points = [False]*len(points)

    for item_s in need_items:
        point_s_loc = [item_s["bbox"][0]+item_s["bbox"][2]/2, item_s["bbox"][1]+item_s["bbox"][3]/2]
        item_nearest = None
        dist_nearest = MAXLEN
        index_nearest = 0
        for index, item_p in enumerate(points):
            if not vist_points[index]:
                point_p_loc = item_p["loc"][0]
                dist = get_point_dist(point_s_loc, point_p_loc)
                if dist < dist_nearest:
                    dist_nearest = dist
                    item_nearest = item_p
                    index_nearest = index
        vist_points[index_nearest] = True
        if not item_nearest is None:
            relations.append([item_s["id"], [item_nearest["id"]]])

def get_textlen2point_relation(elem_dict, symbols, rela_geo2geo, rela_sym2geo):
    """
        nearest neighbor rule -- (midpoint of line) to (text) 
    """
    line_dict = {}
    for item in rela_geo2geo:
        if item[1][0]=='l': 
            if not item[1] in line_dict:
                line_dict[item[1]]=[]
            line_dict[item[1]].append(item[0])

    line_list = []
    for key, value in line_dict.items():
        for point_a, point_b in list(combinations(value, 2)):
            line_list.append([point_a, point_b, key, mid_vec(elem_dict[point_a]['loc'][0], 
                                                            elem_dict[point_b]['loc'][0])])
    need_items  = get_name_item('len', symbols, class_name='text_class')
    vist_lines = [False]*len(line_list)

    for item_s in need_items:
        point_s = [item_s["bbox"][0]+item_s["bbox"][2]/2, item_s["bbox"][1]+item_s["bbox"][3]/2]
        item_nearest = None
        dist_nearest = MAXLEN
        index_nearest = 0
        for index, item_line in enumerate(line_list):
            if not vist_lines[index]:
                dist = get_point_dist(point_s, item_line[3])
                if dist < dist_nearest:
                    dist_nearest = dist
                    item_nearest = item_line
                    index_nearest = index
        if not item_nearest is None:
            vist_lines[index_nearest] = True
            rela_sym2geo.append([item_s["id"],item_nearest[:3]])

def get_textline2line_relation(elem_dict, symbols, rela_geo2geo, rela_sym2geo):

    """
        nearest neighbor rule -- (text) to (point, line)
    """
    point2line_dict = {}
    point_list = []
    for item in rela_geo2geo:
        if item[1][0]=='l' and item[2]=='endpoint': 
            point2line_dict[item[0]]=item[1]
            if not item[0] in point_list:
                point_list.append(item[0])

    need_items  = get_name_item('line', symbols, class_name='text_class')

    for item_s in need_items:
        point_s_loc = [item_s["bbox"][0]+item_s["bbox"][2] /2, item_s["bbox"][1]+item_s["bbox"][3]/2]
        item_nearest = None
        dist_nearest = MAXLEN
        for item_point in point_list:
            dist = get_point_dist(point_s_loc, elem_dict[item_point]['loc'][0])
            if dist < dist_nearest:
                dist_nearest = dist
                item_nearest = item_point
        if not item_nearest is None:
            rela_sym2geo.append([item_s["id"],[item_nearest,point2line_dict[item_nearest]]])

def get_parallel2line_relation(symbols, lines, rela_sym2geo):
    """
        intersection rule -- (symbol) to (line)
    """
    need_items  = get_name_item('parallel', symbols, class_name='sym_class')

    for item_s in need_items:
        item_choose = None
        for item_line in lines:
            # zoom out the symbol box appropriately
            if isLineIntersectRectangle(item_line['loc'][0], item_line['loc'][1], item_s["bbox"], pixes_change=-3):
                item_choose = item_line
        if not item_choose is None:
            rela_sym2geo.append([item_s["id"],[item_choose['id']]])

def get_bar2point_relation(elem_dict, symbols, lines, circles, rela_geo2geo, rela_sym2geo):
    
    """
        intersection rule -- (symbol) to (point, point, line or circle)
    """
    line_dict, circle_dict = {}, {}
    for item in rela_geo2geo:
        if item[1][0]=='l': 
            if not item[1] in line_dict:
                line_dict[item[1]]=[]
            line_dict[item[1]].append([item[0], elem_dict[item[0]]['loc'][0]])
        if item[1][0]=='c': 
            if not item[1] in circle_dict:
                circle_dict[item[1]]=[]
            circle_dict[item[1]].append([item[0], elem_dict[item[0]]['loc'][0]])

    need_items = get_name_item('bar', symbols, class_name='sym_class')

    for item_s in need_items:

        item_choose = None
        point_s_loc = [item_s["bbox"][0]+item_s["bbox"][2] /2, item_s["bbox"][1]+item_s["bbox"][3]/2]

        for item_line in lines:
            if isLineIntersectRectangle(item_line['loc'][0], item_line['loc'][1], item_s["bbox"], pixes_change=-3):
                item_choose = item_line
                break
        if not item_choose is None: 
            point_list = line_dict[item_choose['id']]
            coord_x = [point[1][0] for point in point_list]
            coord_y = [point[1][1] for point in point_list]
            if max(coord_x)-min(coord_x)>max(coord_y)-min(coord_y):
                point_list.sort(key=lambda x:x[1][0])
                for i in range(len(point_list)-1):
                    if point_list[i][1][0]<=point_s_loc[0] and point_list[i+1][1][0]>=point_s_loc[0]:
                        rela_sym2geo.append([item_s["id"],[point_list[i][0], point_list[i+1][0], item_choose['id']]])
                        break
            else:
                point_list.sort(key=lambda x:x[1][1])
                for i in range(len(point_list)-1):
                    if point_list[i][1][1]<=point_s_loc[1] and point_list[i+1][1][1]>=point_s_loc[1]:
                        rela_sym2geo.append([item_s["id"],[point_list[i][0], point_list[i+1][0], item_choose['id']]])
                        break
        else:
            for item_circle in circles:
                item_circle_center = item_circle['loc'][0]
                item_circle_radius = item_circle['loc'][1]
                if abs(get_point_dist(item_circle_center, point_s_loc)-item_circle_radius)<THRESH_DIST*2:
                    item_choose = item_circle
                    break
            if not item_choose is None: 
                point_list = circle_dict[item_choose['id']]
                point_list.sort(key=lambda x: get_point_dist(x[1], point_s_loc))
                if len(point_list)>=2:
                    rela_sym2geo.append([item_s["id"],[point_list[0][0], point_list[1][0], item_choose['id']]])
        
def get_perpendicular_relation(symbols, lines, rela_geo2geo, rela_sym2geo):

    """
        nearest neighbor and degree rule -- (symbol) to (point, line, line)
    """
    line_dict = {}
    for item in rela_geo2geo:
        if item[1][0]=='l': 
            if not item[1] in line_dict:
                line_dict[item[1]]=[]
            line_dict[item[1]].append(item[0])

    need_items = get_name_item('perpendicular', symbols, class_name='sym_class')

    for item_s in need_items:
        item_candiates = []
        for item_line in lines:
            if isLineIntersectRectangle(item_line['loc'][0], item_line['loc'][1], item_s["bbox"], pixes_change=2):
                item_candiates.append(item_line)
        l1, l2 = None, None 
        for line1, line2 in list(combinations(item_candiates, 2)):
            vec1 = sub_vec(line1['loc'][0], line1['loc'][1])
            vec2 = sub_vec(line2['loc'][0], line2['loc'][1])
            if 90-THRESH_DIST*1.5<get_angle_bet2vec(vec1,vec2)<90+THRESH_DIST*1.5:
                l1, l2 = line1, line2
                break 
        if not (l1 is None or l2 is None):
            intersection_list = get_intersection(line_dict[l1['id']], line_dict[l2['id']])
            if len(intersection_list)>0:
                rela_sym2geo.append([item_s["id"],[intersection_list[0], l1['id'], l2['id']]])

def get_angle_relation(elem_dict, symbols, circles, rela_geo2geo, rela_sym2geo, check_name, class_name):
    '''
        position rule (inside the trangle) -- (angle) to (point, line, line) 
    '''
    line_dict, circle_dict = {}, {}
    for item in rela_geo2geo:
        if item[1][0]=='l': 
            if not item[1] in line_dict:
                line_dict[item[1]]=[]
            line_dict[item[1]].append([item[0], elem_dict[item[0]]['loc'][0]])
        if item[1][0]=='c': 
            if not item[1] in circle_dict:
                circle_dict[item[1]]=[]
            circle_dict[item[1]].append([item[0], elem_dict[item[0]]['loc'][0]])

    need_items = get_name_item(check_name, symbols, class_name=class_name)
    candi_angle_list = get_all_angle(line_dict)

    for item_s in need_items:
        candi_item_list = []
        point_center = [item_s["bbox"][0]+item_s["bbox"][2]/2, item_s["bbox"][1]+item_s["bbox"][3]/2]
        for item_angle in candi_angle_list:
            if IsInsideTrangle(elem_dict[item_angle[0]]['loc'][0],
                                elem_dict[item_angle[1]]['loc'][0],
                                elem_dict[item_angle[2]]['loc'][0], 
                                point_center):
                # crosspoint_dist, crosspoint_name, line1_name, line2_name
                len_1 = getDisPointToLine(point_center, elem_dict[item_angle[2]]['loc'][0], elem_dict[item_angle[0]]['loc'][0])
                len_2 = getDisPointToLine(point_center, elem_dict[item_angle[2]]['loc'][0], elem_dict[item_angle[1]]['loc'][0])

                candi_item_list.append([abs(get_point_dist(elem_dict[item_angle[2]]['loc'][0], point_center)),
                                        len_1+len_2, item_angle[2], item_angle[3], item_angle[4]])
        for item_circle in circles: 
            item_circle_center = item_circle['loc'][0]
            item_circle_radius = item_circle['loc'][1]
            if item_circle['id'] in circle_dict:
                point_list = circle_dict[item_circle['id']]
                if len(point_list)>=2:
                    point_list.sort(key=lambda x: get_point_dist(x[1], point_center))
                    candi_item_list.append([3*abs(get_point_dist(item_circle_center, point_center)-item_circle_radius),
                                            0,point_list[0][0],point_list[1][0],item_circle['id']])
                                            
        candi_item_list.sort(key=lambda x: (x[0],x[1]))
        if len(candi_item_list)>0:
            rela_sym2geo.append([item_s["id"], candi_item_list[0][2:]])

def get_relation_rule(prediction, file_name, gt_boxlist, CLASSES_SYM):

    relation_each = dict()
    elem_dict = dict()
    geos = {}
    geo_mask = dict()
    symbols = []
    relations = dict()
    relations['geo2geo'] = []
    relations['sym2geo'] = []
    relations['sym2sym'] = []
    geo_mask['points'] = {}
    geo_mask['lines'] = {}
    geo_mask['circles'] = {}

    relation_each['file_name'] = file_name
    relation_each['width'] = prediction['seg'].size[0]
    relation_each['height'] = prediction['seg'].size[1]
    
    get_geos_pred(geos, prediction['seg'], elem_dict, geo_mask)
    get_geo2geo_relation(geos, elem_dict, geo_mask, prediction['rel'], relations['geo2geo'])

    relation_each['geos'] = geos
    get_syms_pred(symbols, prediction['det'], gt_boxlist, elem_dict, CLASSES_SYM)
    relation_each['symbols'] = symbols
    get_textpoint2point_relation(relation_each['geos']['points'], relation_each['symbols'], relations['sym2geo'])
    get_textlen2point_relation(elem_dict, relation_each['symbols'], relations['geo2geo'], relations['sym2geo'])
    get_textline2line_relation(elem_dict, relation_each['symbols'], relations['geo2geo'], relations['sym2geo'])
    get_parallel2line_relation(relation_each['symbols'], relation_each['geos']['lines'], relations['sym2geo'])
    get_bar2point_relation(elem_dict, relation_each['symbols'], relation_each['geos']['lines'], \
                                    relation_each['geos']['circles'], relations['geo2geo'], relations['sym2geo'])
    get_perpendicular_relation(relation_each['symbols'], relation_each['geos']['lines'], relations['geo2geo'], relations['sym2geo'])

    get_angle_relation(elem_dict, relation_each['symbols'], relation_each['geos']['circles'], \
                                        relations['geo2geo'], relations['sym2geo'], 'angle', 'sym_class')
    get_angle_relation(elem_dict, relation_each['symbols'], relation_each['geos']['circles'], \
                                        relations['geo2geo'], relations['sym2geo'], 'angle', 'text_class')
    get_angle_relation(elem_dict, relation_each['symbols'], relation_each['geos']['circles'], \
                                        relations['geo2geo'], relations['sym2geo'], 'degree', 'text_class')

    relation_each['relations'] = relations

    return relation_each, elem_dict


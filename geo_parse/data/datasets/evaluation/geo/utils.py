import cv2
import torch
import os
import numpy as np
import math
import json

THRESH_DIST = 4
MAXLEN = 1000000000000

def get_point_dist(p0, p1):
    dist = math.sqrt((p0[0]-p1[0])**2
                    +(p0[1]-p1[1])**2)
    return dist

def get_angle_bet2vec(vec1, vec2):
    vector_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    length_prod = get_point_dist([0,0], vec1)*get_point_dist([0,0], vec2)
    cos_value = vector_prod * 1.0 / (length_prod * 1.0 + 1e-8)
    cos_value = min(1,max(-1,cos_value))
    return (math.acos(cos_value) / math.pi) * 180

def sub_vec(vec1,vec2):
    return (np.array(vec1)-np.array(vec2)).tolist()

def mid_vec(vec1,vec2):
    return ((np.array(vec1)+np.array(vec2))/2).tolist()

def get_intersection(list_a, list_b):
    result_list = list(set(list_a)&set(list_b))
    return result_list

def isLineIntersectRectangle(point1, point2, rectangle, pixes_change=0):
    
    linePointX1, linePointY1 = point1[0], point1[1]
    linePointX2, linePointY2 = point2[0], point2[1]
    rectangleLeftTopX, rectangleLeftTopY, rectangleRightBottomX, rectangleRightBottomY = \
        rectangle[0]-pixes_change, rectangle[1]-pixes_change, \
        rectangle[0]+rectangle[2]-1+pixes_change, rectangle[1]+rectangle[3]-1+pixes_change
        
    lineHeight = linePointY1 - linePointY2
    lineWidth = linePointX2 - linePointX1

    c = linePointX1 * linePointY2 - linePointX2 * linePointY1

    if ((lineHeight * rectangleLeftTopX + lineWidth * rectangleLeftTopY + c >= 0 and  
        lineHeight * rectangleRightBottomX + lineWidth * rectangleRightBottomY + c <= 0) or 
        (lineHeight * rectangleLeftTopX + lineWidth * rectangleLeftTopY + c <= 0 and 
        lineHeight * rectangleRightBottomX + lineWidth * rectangleRightBottomY + c >= 0) or 
        (lineHeight * rectangleLeftTopX + lineWidth * rectangleRightBottomY + c >= 0 and
        lineHeight * rectangleRightBottomX + lineWidth * rectangleLeftTopY + c <= 0 )or 
        (lineHeight * rectangleLeftTopX + lineWidth * rectangleRightBottomY + c <= 0 and 
        lineHeight * rectangleRightBottomX + lineWidth * rectangleLeftTopY + c >= 0)):

        if (rectangleLeftTopX > rectangleRightBottomX):
            rectangleLeftTopX, rectangleRightBottomX = rectangleRightBottomX, rectangleLeftTopX

        if (rectangleLeftTopY < rectangleRightBottomY):
            rectangleLeftTopY, rectangleRightBottomY = rectangleRightBottomY, rectangleLeftTopY 
 
        if ((linePointX1 < rectangleLeftTopX and linePointX2 < rectangleLeftTopX)
            or (linePointX1 > rectangleRightBottomX and linePointX2 > rectangleRightBottomX)
            or (linePointY1 > rectangleLeftTopY and linePointY2 > rectangleLeftTopY)
            or (linePointY1 < rectangleRightBottomY and linePointY2 < rectangleRightBottomY)):
            return False
        else:
            return True
    else:
        return False
        
def IsTrangleOrArea(x1,y1,x2,y2,x3,y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
 
def IsInsideTrangle(trangle_point1, trangle_point2, trangle_point3, judge_point):
    thresh_value= 2.0
    increase_ratio = 3.0
    x1, y1 = trangle_point1[0], trangle_point1[1]
    x2, y2 = trangle_point2[0], trangle_point2[1]
    x3, y3 = trangle_point3[0], trangle_point3[1] # intersection point
    x, y = judge_point[0], judge_point[1] # judge point
    # enlarge the sides of triangles
    x1, y1 = increase_ratio*x1+(1-increase_ratio)*x3, increase_ratio*y1+(1-increase_ratio)*y3
    x2, y2 = increase_ratio*x2+(1-increase_ratio)*x3, increase_ratio*y2+(1-increase_ratio)*y3
    ABC = IsTrangleOrArea(x1,y1,x2,y2,x3,y3)
    PBC = IsTrangleOrArea(x,y,x2,y2,x3,y3)
    PAC = IsTrangleOrArea(x1,y1,x,y,x3,y3)
    PAB = IsTrangleOrArea(x1,y1,x2,y2,x,y)
    return abs(ABC-PBC-PAC-PAB)<thresh_value

def get_angle_point(relation, elem_dict, line_dict, img_name=None, text_class_name='angle', text_content=None):
    
    """
    Determine the direction of the angle according to the primitive in the triangle, and return the nearest neighbor points  
    special case:
        1. the side of angle passes through the rectangular box of primitive (it happens with head at some cases), 
           then select the angle with similar degree  
        2. degree of angle is large and the triangle still cannot include the center of primitive after enlargement, 
           then choose the only reasonable angle  
        3. the two corresponding edges are the same (degree is 180), select the endpoint of the edge
    """
    sym = relation[0]
    sym_bbox = elem_dict[sym]['bbox']
    sym_center_loc = [sym_bbox[0]+sym_bbox[2]/2, sym_bbox[1]+sym_bbox[3]/2]

    crosspoint = relation[1][0]
    crosspoint_loc = elem_dict[crosspoint]['loc'][0]

    l1_point_list, l2_point_list = line_dict[relation[1][1]], line_dict[relation[1][2]]
    point_num_l1, point_num_l2 = len(l1_point_list), len(l2_point_list)

    crosspoint_l1_index, crosspoint_l2_index = None, None 

    for index in range(point_num_l1):
        if l1_point_list[index][0]==crosspoint:
            crosspoint_l1_index = index 
            break
    for index in range(point_num_l2):
        if l2_point_list[index][0]==crosspoint:
            crosspoint_l2_index = index 
            break
    
    if crosspoint_l1_index is None or crosspoint_l2_index is None: 
        return None, None

    angle_num = 0
    near_p1, near_p2 = crosspoint_l1_index, crosspoint_l2_index

    for point_l1_index in [0, point_num_l1-1]:
        if point_l1_index!=crosspoint_l1_index:
            for point_l2_index in [0, point_num_l2-1]:
                if point_l2_index!=crosspoint_l2_index:
                    angle_num += 1 
                    t1, t2 = point_l1_index, point_l2_index
                    if IsInsideTrangle(l1_point_list[point_l1_index][1], l2_point_list[point_l2_index][1], \
                                                crosspoint_loc, sym_center_loc):
                        if point_l1_index==0: 
                            near_p1-=1 
                        else:
                            near_p1+=1
                        if point_l2_index==0: 
                            near_p2-=1 
                        else:
                            near_p2+=1

    if near_p1!=crosspoint_l1_index and near_p2!=crosspoint_l2_index:
        if not text_content is None:
            text_content = text_content.replace(' ', '').split('^')[0]
        if elem_dict[sym]['sym_class']=='head' and text_class_name=='degree' and text_content.isdigit(): # case 1
            degree_gt= int(text_content)
            angle_degree = get_angle_bet2vec(sub_vec(l1_point_list[near_p1][1], crosspoint_loc), \
                                sub_vec(l2_point_list[near_p2][1], crosspoint_loc))
            if abs(degree_gt-angle_degree)>20:
                if isLineIntersectRectangle(crosspoint_loc, l1_point_list[near_p1][1], sym_bbox):
                    near_p2 = 2*crosspoint_l2_index-near_p2
                else:
                    near_p1 = 2*crosspoint_l1_index-near_p1
        return l1_point_list[near_p1], l2_point_list[near_p2]   
    elif angle_num==1:  # case 2
        if t1==0: 
            near_p1-=1 
        else:
            near_p1+=1
        if t2==0: 
            near_p2-=1 
        else:
            near_p2+=1
        return l1_point_list[near_p1], l2_point_list[near_p2]
    elif relation[1][1]==relation[1][2]: # case 3
        return l1_point_list[0], l1_point_list[-1]

    else:
        return None, None

def get_elem_dict(primitives):
    elem_dict=dict()
    for prim_item in primitives:
        for item in prim_item:
            elem_dict[item['id']] = item
    return elem_dict

def get_annotation_json(path_file):
    with open(path_file, 'rb') as file:
        contents = json.load(file)
    return contents

def get_name_item(name, symbols, class_name='sym_class'):

    need_items = []
    for item_s in symbols:
        if name in item_s[class_name]:
            need_items.append(item_s)

    return need_items

def get_name_id(name, symbols, class_name='sym_class'):
    
    id_items = []
    for item_s in symbols:
        if name in item_s[class_name]:
            id_items.append(item_s['id'])
    return id_items

def get_connected_node(id_src_node, rel_map, CLASSES_SEM, name_list=None):
    
    if name_list is None: name_list = CLASSES_SEM
    x = rel_map.id2nodeid[id_src_node]
    node_list = []

    for y in range(len(rel_map.ids)):
        name = CLASSES_SEM[rel_map.labels_sem[y]]
        if name in name_list and rel_map.edge_label[x,y]==1 and rel_map.edge_label[y, x]==1:
            node_list.append({'id':rel_map.ids[y], 'name':name})

    # sort nodes by points, lines, circles, and symbols
    seq_id_dict = {'p':0,'l':1,'c':2,'s':3}
    node_list.sort(key=lambda x:seq_id_dict[x['id'][0]])
    id_node_list = [item['id'] for item in node_list]
    name_node_list = [item['name'] for item in node_list]

    return name_node_list, id_node_list

def is_meet_cond(name_node_list, name_list):
    o = True
    if len(name_node_list)!=len(name_list):
        o=False
    else:
        for node_name, name in zip(name_node_list, name_list):
            if node_name!=name:
                o=False
                break
    return o

def get_head_len2point(elem_dict, id_head_len_list, id_point_list):
    '''
        find the point corresponding to head_len according to distance and lenth of line
    '''
    head_len_locs = []
    cand_point_locs = []
    cand_point_ids = []
    point_locs = []
    thresh_ratio = 0.15

    for id_head in id_head_len_list:
        sym_bbox = elem_dict[id_head]['bbox']
        head_len_center = [sym_bbox[0]+sym_bbox[2]/2, sym_bbox[1]+sym_bbox[3]/2]
        head_len_locs.append(head_len_center)
        incre_dist = (sym_bbox[2]+sym_bbox[3])/2
    for id_point in id_point_list:
        point_locs.append(elem_dict[id_point]['loc'][0])

    for i in range(len(id_head_len_list)):
        dist_record = []
        for j in range(len(id_point_list)):
            dist_record.append(get_point_dist(head_len_locs[i], point_locs[j]))
        t = dist_record.index(min(dist_record))
        cand_point_ids.append(id_point_list[t])
        cand_point_locs.append(point_locs[t])  

    dist_head_len = get_point_dist(head_len_locs[0], head_len_locs[1])+incre_dist
    dist_cand_point = get_point_dist(cand_point_locs[0], cand_point_locs[1]) 
    dist_head_len2cand_point = [get_point_dist(item1, item2) for item1, item2 in zip(head_len_locs, cand_point_locs)]

    if abs(dist_cand_point/dist_head_len-1)<thresh_ratio and \
        abs(dist_head_len2cand_point[0]/dist_head_len2cand_point[1]-1)<thresh_ratio:
        return cand_point_ids
    
    cand_item_list = []
    for i in range(len(id_point_list)):
        for j in range(len(id_point_list)):
            if i!=j:
                dist1 = get_point_dist(point_locs[i], point_locs[j]) 
                dist2 = get_point_dist(point_locs[i], head_len_locs[0]) 
                dist3 = get_point_dist(point_locs[j], head_len_locs[1]) 
                cand_item_list.append([(id_point_list[i],id_point_list[j]), \
                                        abs(dist1/dist_head_len-1)+abs(dist2/dist3-1)])

    cand_item_list.sort(key=lambda x:x[1])
    if len(cand_item_list)>0 and cand_item_list[0][1]<thresh_ratio:
        return cand_item_list[0][0]
    else:
        return []

def get_all_angle(line_dict):
    '''
        get all possible angle from line set
    '''
    candi_angle_list = []

    for id in line_dict:
        point_list = line_dict[id]
        coord_x = [point[1][0] for point in point_list]
        coord_y = [point[1][1] for point in point_list]
        if max(coord_x)-min(coord_x)>max(coord_y)-min(coord_y):
            line_dict[id].sort(key=lambda x:x[1][0])
        else:
            line_dict[id].sort(key=lambda x:x[1][1])

    key_list = list(line_dict.keys())

    for idx in range(len(key_list)-1):
        point_list_x = [x[0] for x in line_dict[key_list[idx]]]
        for idy in range(idx+1, len(key_list)):
            point_list_y = [y[0] for y in line_dict[key_list[idy]]]
            crosspoint_list = get_intersection(point_list_x, point_list_y)
            if len(crosspoint_list)==1:
                crosspoint = crosspoint_list[0]
                for idx_endpoint in [0, -1]:
                    if point_list_x[idx_endpoint] != crosspoint:
                        for idy_endpoint in [0, -1]:
                            if point_list_y[idy_endpoint] != crosspoint:
                                candi_angle_list.append([point_list_x[idx_endpoint], 
                                                        point_list_y[idy_endpoint], 
                                                        crosspoint,
                                                        key_list[idx],
                                                        key_list[idy]]
                                                        )
    return candi_angle_list
                
def getFootPoint(point, line_p1, line_p2):
    """
        point, line_p1, line_p2 : [x, y, z]
    """
    x0 = point[0]
    y0 = point[1]
    z0 = 0 # point[2]
    x1 = line_p1[0]
    y1 = line_p1[1]
    z1 = 0 # line_p1[2]
    x2 = line_p2[0]
    y2 = line_p2[1]
    z2 = 0 # line_p2[2]

    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * 1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    zn = 0 # k * (z2 - z1) + z1

    return (xn, yn)

def getDisPointToLine(point, line_p1, line_p2):

    footP = getFootPoint(point, line_p1, line_p2)
    
    return get_point_dist(footP, point) 

def show_img_result(img, file_name, CLASSES_SYM, CLASSES_GEO, prediction, output_im_dir, id2name_dict=None, CLASSES_TEXT=None):

    img = cv2.resize(img, prediction['seg'].size)
    # img_geo[0]: point, img_geo[1]: line, img_geo[2]: circle
    img_geo = [(img.copy()*0.2).astype(np.uint8) for _ in range(len(CLASSES_GEO)-1)]

    ###################### detection ############################
    boxes = prediction['det'].bbox
    labels_det = [CLASSES_SYM[i] for i in prediction['det'].get_field("labels")]
    if not CLASSES_TEXT is None:
        labels_text = [CLASSES_TEXT[i] for i in prediction['det'].get_field("labels_text")]
    for index, (box, label) in enumerate(zip(boxes, labels_det)):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        cv2.rectangle(
            img, tuple(top_left), tuple(bottom_right), color=(0, 255, 0), thickness=1)
        x, y = box[:2]
        if not CLASSES_TEXT is None and label=='text':
            label = label+"_"+labels_text[index]
        cv2.putText(img, label, (int(x.item()), int(y.item())), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)

    ######################## segmentation ###########################
    masks = prediction['seg'].masks
    inst_num = [0,0,0]
    for index, (class_index, loc, id) in enumerate(zip(prediction['seg'].get_field("labels"), \
                                    prediction['seg'].get_field("locs"), prediction['seg'].get_field("ids"))):
        color = np.random.randint(0,256,size=[3])
        img_geo[class_index-1][masks[:,:,index]>0] = color
        inst_num[class_index-1]+=1

        if not id2name_dict is None:
            if class_index==1 or class_index==3:
                point_name = id2name_dict[id]
                point_loc = (round(loc[0][0]), round(loc[0][1]))
                if class_index==1:
                    cv2.putText(img_geo[0], point_name, point_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(255, 0, 0), thickness=2)
                else:
                    cv2.putText(img_geo[2], point_name, point_loc, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(0, 0, 255), thickness=2)

    cv2.putText(img_geo[0], str(inst_num[0]), (0, img_geo[0].shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
    cv2.putText(img_geo[1], str(inst_num[1]), (0, img_geo[1].shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
    cv2.putText(img_geo[2], str(inst_num[2]), (0, img_geo[2].shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

    row1_img = np.concatenate((img, img_geo[0]), axis=1)
    row2_img = np.concatenate((img_geo[1], img_geo[2]), axis=1)
    img_draw = np.concatenate((row1_img, row2_img), axis=0)

    cv2.imwrite(os.path.join(output_im_dir, file_name), img_draw) 


def show_img_result_demo(img, CLASSES_GEO, prediction, id2name_dict=None):
        
    img = cv2.resize(img, prediction['seg'].size)
    # img_geo[0]: point, img_geo[1]: line, img_geo[2]: circle
    img_geo = [(img.copy()*0.2).astype(np.uint8) for _ in range(len(CLASSES_GEO)-1)]
    
    ###################### detection image ############################
    for index, (box, id) in enumerate(zip(prediction['det'].bbox, prediction['det'].get_field("ids"))):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        cv2.rectangle(img, tuple(top_left), tuple(bottom_right), color=(0, 255, 0), thickness=1)
        x, y = box[:2]
        cv2.putText(img, id, (int(x.item()), int(y.item())), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
    ######################## segmentation image ###########################
    masks = prediction['seg'].masks
    inst_num = [0,0,0]
    for index, (class_index, loc, id) in enumerate(zip(prediction['seg'].get_field("labels"), \
                                    prediction['seg'].get_field("locs"), prediction['seg'].get_field("ids"))):
        color = np.random.randint(0,256,size=[3])
        img_geo[class_index-1][masks[:,:,index]>0] = color
        inst_num[class_index-1]+=1

        if not id2name_dict is None:
            if class_index==1 or class_index==3:
                point_name = id2name_dict[id]
                point_loc = (round(loc[0][0]), round(loc[0][1]))
                if class_index==1:
                    cv2.putText(img_geo[0], point_name, point_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(255, 0, 0), thickness=2)
                else:
                    cv2.putText(img_geo[2], point_name, point_loc, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(0, 0, 255), thickness=2)

    return img, img_geo[0], img_geo[1], img_geo[2] 
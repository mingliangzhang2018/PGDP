import os
import cv2
import matplotlib.pyplot as plt
from use_geosolver import image_to_graph_parse
import numpy as np
from geosolver.diagram.get_instances import get_all_instances
from generate_logic_form import calc_angle, dist
import math
from tqdm import tqdm
from multiprocess import Pool
import json
import shutil


def gen_json(f, height, width):
    json_dict = dict()
    json_dict["version"] = "4.5.9"
    json_dict["flags"] = {}
    json_dict["shapes"] = []
    json_dict["imagePath"] = f
    json_dict["imageData"] = None
    json_dict["imageHeight"] = height
    json_dict["imageWidth"] = width
    return json_dict


def get_lines(graph_parse, offset, json_dict):

    lines = get_all_instances(graph_parse, 'line')
    l = len(lines)
    is_max_line = [True]*l
    line_candiates = [{"key": key, "x1": value.a.x + offset.x,
                       "y1": value.a.y + offset.y, "x2": value.b.x + offset.x,
                       "y2": value.b.y + offset.y} for key, value in lines.items()]
    
    for i in range(l):
        line_candiates[i]['len'] = dist((line_candiates[i]['x1'], line_candiates[i]['y1']),
                                        (line_candiates[i]['x2'], line_candiates[i]['y2']))

    line_candiates = sorted(line_candiates, key=lambda x: x['len'])

    for i in range(l):
        if line_candiates[i]['len'] < 50:
            is_max_line[i] = False

    for i in range(l-1):
        if is_max_line[i]:
            for j in range(i+1, l):
                if is_max_line[j]:
                    point_i = line_candiates[i]['key']
                    point_j = line_candiates[j]['key']
                    if len(set(point_i+point_j)) < 4:
                        # print(line_candiates[i])
                        # print(line_candiates[j])
                        if (line_candiates[j]['x1'] != line_candiates[i]['x1'] or line_candiates[j]['y1'] != line_candiates[i]['y1']) and \
                                (line_candiates[j]['x1'] != line_candiates[i]['x2'] or line_candiates[j]['y1'] != line_candiates[i]['y2']):
                            angle = calc_angle((line_candiates[i]['x1'], line_candiates[i]['y1']),
                                               (line_candiates[i]['x2'],
                                                line_candiates[i]['y2']),
                                               (line_candiates[j]['x1'], line_candiates[j]['y1']))
                        else:
                            angle = calc_angle((line_candiates[i]['x1'], line_candiates[i]['y1']),
                                               (line_candiates[i]['x2'],
                                                line_candiates[i]['y2']),
                                               (line_candiates[j]['x2'], line_candiates[j]['y2']))
                        # The angle needs to approach pi (180).
                        if angle >= math.pi - 0.1 or angle <= 0.1:
                            is_max_line[i] = False
                            break
    line_num = 0
    for i in range(l):
        if is_max_line[i]: 
            line_dict = dict()
            line_dict['label']="line_"+str(line_num)
            line_dict['points']=[[line_candiates[i]["x1"],line_candiates[i]["y1"]],
                                    [line_candiates[i]["x2"],line_candiates[i]["y2"]]]
            line_dict['group_id']=None
            line_dict['shape_type']='line'
            line_dict['flags']={}

            json_dict["shapes"].append(line_dict)
            line_num+=1


def get_circles(graph_parse, offset, json_dict):

    circles = get_all_instances(graph_parse, 'circle')
    circle_num = 0
    for _, value in circles.items():
        x = value.center.x + offset.x
        y = value.center.y + offset.y
        radius = value.radius
        circle_dict = dict()
        circle_dict['label']="circle_"+str(circle_num)
        circle_dict['points']=[[x,y], [x+radius,y]]
        circle_dict['group_id']=None
        circle_dict['shape_type']='circle'
        circle_dict['flags']={}

        json_dict["shapes"].append(circle_dict)
        circle_num+=1

def get_points(graph_parse, offset, json_dict):
     
    points = get_all_instances(graph_parse, 'point')
    point_num = 0
    for _, value in points.items():
        x = value.x + offset.x
        y = value.y + offset.y

        point_dict = dict()
        point_dict['label']="point_"+str(point_num)
        point_dict['points']=[[x,y]]
        point_dict['group_id']=None
        point_dict['shape_type']='point'
        point_dict['flags']={}

        json_dict["shapes"].append(point_dict)
        point_num+=1


def get_primitives(parameters):

    path_ori, f, path_save_point, path_save_circle = parameters
    img_path = os.path.join(path_ori, f)
    # img_path = '/lustre/home/mlzhang/GeoMathQA/InterGPS/data/geometry3k/No Duplicates/1147.png'
    # img_path_save = os.path.join(path_save, f)
    img = cv2.imread(img_path)

    json_dict_point = gen_json(f, img.shape[0], img.shape[1])
    json_dict_circle = gen_json(f, img.shape[0], img.shape[1])

    try:
        graph_parse = image_to_graph_parse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        offset = graph_parse.image_segment_parse.diagram_image_segment.offset

        get_circles(graph_parse, offset, json_dict_circle)
        # get_lines(graph_parse, offset, json_dict)
        get_points(graph_parse, offset, json_dict_point)


    except:
        print(f)

    with open(os.path.join(path_save_point, f[:-3]+"json"), "w") as json_file:
        json.dump(json_dict_point, json_file, indent=2)
    
    with open(os.path.join(path_save_circle, f[:-3]+"json"), "w") as json_file:
        json.dump(json_dict_circle, json_file, indent=2)


path_ori = "/lustre/home/mlzhang/GeoMathQA/InterGPS/data/geometry3k/No Duplicates"

path_save_point = "/lustre/home/mlzhang/GeoMathQA/InterGPS/diagram_parser/parser/json_gen_point"
if os.path.exists(path_save_point):
    shutil.rmtree(path_save_point)
os.makedirs(path_save_point)

path_save_circle = "/lustre/home/mlzhang/GeoMathQA/InterGPS/diagram_parser/parser/json_gen_circle"
if os.path.exists(path_save_circle):
    shutil.rmtree(path_save_circle)
os.makedirs(path_save_circle)

files = os.listdir(path_ori)

para_lst = [(path_ori, file, path_save_point, path_save_circle) for file in files]

with tqdm(total=len(files), ncols=100) as t:  # ncols 可以自定义进度条的总长度
    with Pool(24) as p:  # 32个进程
        for _ in p.imap_unordered(get_primitives, para_lst):
            # solve_list.append(answer)
            t.update()

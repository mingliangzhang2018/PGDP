from .utils import *
from itertools import combinations

def get_point_name(points, circles, textpoint2point, point2circle, 
                    elem_dict, id2name_dict):
    '''
        points: point_list
        circles: circle_list
        textpoint2point: [s0, [p0]]
        point2circle: relation [p0, c0, oncircle/center]
        id2name_dict: {id -> point_name}
        position_dict: {point_name -> loc_list}
    '''
    point_instances = list() # point name list
    point_positions = dict() # point name -> loc
    circle_instances = list() # circle name list

    for item in textpoint2point:
        sym_id, geo_id = item[0], item[1][0]  # s0, p0
        id2name_dict[geo_id] = elem_dict[sym_id]['text_content']
        point_instances.append(id2name_dict[geo_id]) 
        point_positions[id2name_dict[geo_id]] =  elem_dict[geo_id]['loc'][0]

    # generate letter candidates for representing the names of point
    candidate_point_name = list(set([chr(i) for i in range(65,91)]) - set(point_instances)) 
    candidate_point_name.sort()
    index = 0

    for item in points:
        geo_id = item['id'] 
        if not geo_id in id2name_dict: 
            id2name_dict[geo_id] = candidate_point_name[index]
            point_instances.append(id2name_dict[geo_id])
            point_positions[id2name_dict[geo_id]] =  elem_dict[geo_id]['loc'][0]
            index +=1
    
    circle_id2center_id = dict() # circle id -> center id
    circle_name_num = dict() # count of concentric circles
    circle_index = dict() # index of concentric circles

    for item in point2circle:
        if item[2]=='center':
            circle_name_num[item[0]] = circle_name_num.get(item[0],0) + 1
            circle_index[item[0]] = 0

    for item in point2circle:
        if item[2]=='center':
            circle_id2center_id[item[1]] = item[0]
            circle_name = id2name_dict[item[0]]
            # same letter represents multiple concentric circles, denoted as name+index
            if circle_name_num[item[0]]>1:
                circle_name = circle_name+str(circle_index[item[0]])
                circle_index[item[0]]+=1 
            id2name_dict[item[1]] = circle_name
            circle_instances.append(circle_name)

    for item in circles:
        geo_id = item['id']
        if not geo_id in circle_id2center_id:
            id2name_dict[geo_id] = candidate_point_name[index]
            point_instances.append(candidate_point_name[index])
            circle_instances.append(candidate_point_name[index])
            point_positions[candidate_point_name[index]] = elem_dict[geo_id]['loc'][0]
            index +=1
            
    return point_instances, point_positions, circle_instances

def get_line_instances(lines, point2line, elem_dict, id2name_dict, line_dict):
    '''
        point2line: relation [p0, l0, endpoint/online]
        line_dict: line id -> [point id, point loc]
    '''
    line_instances = []

    for item in point2line:
        if not item[1] in line_dict:
            line_dict[item[1]]=[]
        line_dict[item[1]].append([item[0], elem_dict[item[0]]['loc'][0]])

    # the points on the line are sorted along the x or y axes (dimensions with large variances)
    for item in lines:
        point_list = line_dict[item['id']]
        coord_x = [point[1][0] for point in point_list]
        coord_y = [point[1][1] for point in point_list]
        if max(coord_x)-min(coord_x)>max(coord_y)-min(coord_y):
            line_dict[item['id']].sort(key=lambda x:x[1][0])
        else:
            line_dict[item['id']].sort(key=lambda x:x[1][1])
        for point_a, point_b in list(combinations(line_dict[item['id']], 2)):
            if not (id2name_dict[point_a[0]] is None or  id2name_dict[point_b[0]] is None):
                line_instances.append(id2name_dict[point_a[0]]+id2name_dict[point_b[0]])
    
    return line_instances

def get_PointLiesOnLine(id2name_dict, line_dict, image_id):

    list_PointLiesOnLine = list()
    for _,point_list in line_dict.items():
        for k in range(1,len(point_list)-1):
            for v in range(k):
                for l in range(k+1,len(point_list)):
                    try:
                        logic_form_each = "PointLiesOnLine("+ id2name_dict[point_list[k][0]]+", Line("+ \
                                id2name_dict[point_list[v][0]]+", "+id2name_dict[point_list[l][0]]+"))"
                    except:
                        print(image_id, "error in PointLiesOnLine!")
                        continue
                    list_PointLiesOnLine.append(logic_form_each)
    return list_PointLiesOnLine

def get_PointLiesOnCircle(elem_dict, id2name_dict, point2circle, circles, circle_dict, image_id):

    list_PointLiesOnCircle = list()
    for item in circles:
        circle_dict[item['id']] = [] 
    for item in point2circle:
        if item[2]!='center':
            try:
                logic_form_each = "PointLiesOnCircle("+id2name_dict[item[0]]+", Circle(" \
                                + id2name_dict[item[1]]+", radius_"+item[1][1:]+"_0))"
                # TODO 
                # sort the points on circle according to their positions
            except:
                print(image_id, "error in PointLiesOnCircle!")
                continue
            circle_dict[item[1]].append([item[0], elem_dict[item[0]]['loc'][0]]) 
            list_PointLiesOnCircle.append(logic_form_each)
    return list_PointLiesOnCircle

def get_Parallel(id2name_dict, line_dict, parallel, elem_dict, img_name):
    
    list_Parallel2Line = list()
    parallel_list = []
    double_parallel_list = []
    triple_parallel_list = []

    def get_logic_form(item_list):
        choice_items = []
        for i in range(len(item_list)-1):
            for j in range(i+1, len(item_list)):
                if item_list[i]!=item_list[j] and not [item_list[i], item_list[j]] in choice_items:
                    line1_point_list = line_dict[item_list[i]]
                    line2_point_list = line_dict[item_list[j]]
                    line1_endpoint1 = id2name_dict[line1_point_list[0][0]]
                    line1_endpoint2 = id2name_dict[line1_point_list[-1][0]]
                    line2_endpoint1 = id2name_dict[line2_point_list[0][0]] 
                    line2_endpoint2 = id2name_dict[line2_point_list[-1][0]]
                    try:
                        logic_form_each = "Parallel(Line("+line1_endpoint1+", "+line1_endpoint2+"), Line(" \
                                            +line2_endpoint1+", "+line2_endpoint2+"))"
                    except:
                        print(img_name, "error in Parallel!")
                        continue
                    list_Parallel2Line.append(logic_form_each)
                    choice_items.append([item_list[i], item_list[j]])
                    choice_items.append([item_list[j], item_list[i]])
        
    for item in parallel:

        if not (len(item)==2 and item[1][0][0]=='l'):
            print(img_name+" relation of parallel symbol does not conform to standard composition !")
            continue

        if elem_dict[item[0]]['sym_class']=='parallel':
            parallel_list.append(item[1][0])
        if elem_dict[item[0]]['sym_class']=='double parallel':
            double_parallel_list.append(item[1][0])
        if elem_dict[item[0]]['sym_class']=='triple parallel':
            triple_parallel_list.append(item[1][0])

    if len(parallel_list)>0: get_logic_form(parallel_list)
    if len(double_parallel_list)>0: get_logic_form(double_parallel_list)
    if len(triple_parallel_list)>0: get_logic_form(triple_parallel_list)

    return list_Parallel2Line

def get_Perpendicular(id2name_dict, line_dict, perpendicular, elem_dict, img_name):

    def get_near_point(line, line_dict, crosspoint, point_loc):
        line_point_list = line_dict[line]
        point_index = -1
        for i in range(len(line_point_list)):
            if line_point_list[i][0]==crosspoint:
                point_index = i
                break
        if point_index==-1:
            print(img_name+" perpendicular symbol has no intersection!")
            
            return None
        if point_index==0: # the intersection is the endpoint
            point = line_point_list[1][0]
        elif point_index==len(line_point_list)-1: # the intersection is the endpoint
            point = line_point_list[-2][0]
        elif get_angle_bet2vec(sub_vec(center, point_loc), sub_vec(line_point_list[point_index-1][1], point_loc))<90:
            # select a point in the same direction as the perpendicular symbol
            point = line_point_list[point_index-1][0]
        else:
            point = line_point_list[point_index+1][0]
        return point

    list_Perpendicular = list()
    for item in perpendicular:
        if not (len(item[1])==3 and item[1][0][0]=='p' and item[1][1][0]=='l' and item[1][2][0]=='l'):
            print(img_name+" relation of perpendicular symbol does not conform to standard composition !")
            continue
        center = [elem_dict[item[0]]["bbox"][0] + elem_dict[item[0]]["bbox"][2]/2,
                    elem_dict[item[0]]["bbox"][1] + elem_dict[item[0]]["bbox"][3]/2]
        point = item[1][0]
        line1 = item[1][1]
        line2 = item[1][2]
        point_loc = elem_dict[point]['loc'][0]
        try:
            point1 = get_near_point(line1, line_dict, point, point_loc)
            point2 = get_near_point(line2, line_dict, point, point_loc)
            logic_form_each = "Perpendicular(Line("+id2name_dict[point1]+", "+id2name_dict[point]+"), Line(" \
                                    +id2name_dict[point2]+", "+id2name_dict[point]+"))"
        except:
            print(img_name, 'error in Perpendicular!')
            continue                            
        list_Perpendicular.append(logic_form_each)

    return list_Perpendicular

def get_Textlen(id2name_dict, textlen, elem_dict, sym2head_dict, circles, img_name):
    '''
        the relation form of length text 
        1. [p, p, l]
        2. [p, p, c]
        3. [p, p]
        4. [] only consider the situation of circle diameter
    '''
    list_Textlen = list()
    for item in textlen:
        sym = item[0]
        content = elem_dict[sym]['text_content']
        logic_form_each = None
        try:
            if len(item[1])==3:
                if not (item[1][0][0]=='p' and item[1][1][0]=='p' and (item[1][2][0]=='l' or item[1][2][0]=='c')):
                    print(img_name+" relation of length text does not conform to standard composition !")
                    
                    continue       
                point1 = item[1][0]
                point2 = item[1][1]
                if item[1][2][0]=='l':
                    logic_form_each = "Equals(LengthOf(Line("+id2name_dict[point1]+', '+ \
                                        id2name_dict[point2]+")), "+content+")"
                else:
                    logic_form_each = "Equals(LengthOf(Arc("+id2name_dict[point1]+', '+ \
                                        id2name_dict[point2]+")), "+content+")"
            elif len(item[1])==2:
                if not (item[1][0][0]=='p' and item[1][1][0]=='p'):
                    print(img_name+" relation of length text does not conform to standard composition !")
                    continue 
                point1 = item[1][0]
                point2 = item[1][1]
                logic_form_each = "Equals(LengthOf(Line("+id2name_dict[point1]+', '+ \
                                        id2name_dict[point2]+")), "+content+")"
            elif len(item[1])==0:
                if len(circles)>0 and (sym in sym2head_dict) and len(sym2head_dict[sym])==2:
                    for item in circles:
                        diameter = item['loc'][1]*2
                        head1_bbox = elem_dict[sym2head_dict[sym][0]]["bbox"]
                        head2_bbox = elem_dict[sym2head_dict[sym][1]]["bbox"]
                        head1_center = [head1_bbox[0]+head1_bbox[2]/2, head1_bbox[1]+head1_bbox[3]/2]
                        head2_center = [head2_bbox[0]+head2_bbox[2]/2, head2_bbox[1]+head2_bbox[3]/2]
                        if abs(get_point_dist(head1_center, head2_center)/diameter-1)<0.13:
                            logic_form_each = "Equals(PerimeterOf(Circle("+id2name_dict[item['id']]+")), "+content+")"
                            break
        except:
            print(img_name, "error in Length!")
            continue

        if not logic_form_each is None:
            list_Textlen.append(logic_form_each)
        else:
                print(img_name+' can not generate formal language of text len !')

    return list_Textlen

def get_Bar(id2name_dict, bar, elem_dict, img_name):

    def get_logic_form(item_list, list_Bar):
        for i in range(len(item_list)-1):  
            for j in range(i+1, len(item_list)): 
                try:
                    logic_form_each = "Equals("
                    line1_point1, line1_point2 = id2name_dict[item_list[i][0]], id2name_dict[item_list[i][1]]
                    if item_list[i][2][0]=='l':
                        logic_form_each+='LengthOf(Line('+line1_point1+", "+line1_point2+")), "
                    elif item_list[i][2][0]=='c':
                        logic_form_each+='LengthOf(Arc('+line1_point1+", "+line1_point2+")), "
                    line2_point1, line2_point2 = id2name_dict[item_list[j][0]], id2name_dict[item_list[j][1]]
                    if item_list[j][2][0]=='l':
                        logic_form_each+='LengthOf(Line('+line2_point1+", "+line2_point2+")))"
                    elif item_list[j][2][0]=='c':
                        logic_form_each+='LengthOf(Arc('+line2_point1+", "+line2_point2+")))"
                except:
                    print(img_name, 'error in bar!')

                list_Bar.append(logic_form_each)

    list_Bar = list()
    bar_list, double_bar_list, triple_bar_list, quad_bar_list  = [], [], [], []

    for item in bar:
        if not (len(item[1])==3 and item[1][0][0]=='p' and item[1][1][0]=='p' and (item[1][2][0]=='l' or item[1][2][0]=='c')):
            print(img_name+" relation of bar symbol does not conform to standard composition !")
            continue         
        if elem_dict[item[0]]['sym_class']=='bar':
            bar_list.append(item[1])
        if elem_dict[item[0]]['sym_class']=='double bar':
            double_bar_list.append(item[1])
        if elem_dict[item[0]]['sym_class']=='triple bar':
            triple_bar_list.append(item[1])
        if elem_dict[item[0]]['sym_class']=='quad bar':
            quad_bar_list.append(item[1])

    if len(bar_list)>0: get_logic_form(bar_list, list_Bar)
    if len(double_bar_list)>0: get_logic_form(double_bar_list, list_Bar)
    if len(triple_bar_list)>0: get_logic_form(triple_bar_list, list_Bar)
    if len(quad_bar_list)>0: get_logic_form(quad_bar_list, list_Bar)

    return list_Bar

def get_Symangle(id2name_dict, symangle, elem_dict, line_dict, img_name):

    def get_logic_form(item_list, list_angle):
        item_new_list = []
        for i in range(len(item_list)):
            crosspoint = item_list[i][1][0]
            crosspoint_loc = elem_dict[crosspoint]['loc'][0]
            # get the nearest neighbor points on two sides of angle
            point1, point2 = get_angle_point(item_list[i], elem_dict, line_dict) 
            if point1==None or point2==None:
                print(img_name+" no corresponding angle for angle symbol !")
                continue
            angle_degree = get_angle_bet2vec(sub_vec(point1[1], crosspoint_loc), sub_vec(point2[1], crosspoint_loc))
            # add neighbor points and angles
            item_new_list.append([crosspoint, [item_list[i][1][1], point1[0]], [item_list[i][1][2], point2[0]], angle_degree]) 
        # merge neighbor angles 
        item_merge_list = []
        is_merge = [False]*len(item_new_list)
        for i in range(len(item_new_list)):
            if not is_merge[i]:
                for j in range(i+1, len(item_new_list)):
                    if not is_merge[j] \
                        and (item_new_list[i][0]==item_new_list[j][0]) \
                            and abs(item_new_list[i][3]-item_new_list[j][3])>10:
                        is_merge[j] = True
                        item_new_list[i][3] = item_new_list[i][3] + item_new_list[j][3]
                        if  item_new_list[i][1][0]==item_new_list[j][1][0]:
                            item_new_list[i][1] = item_new_list[j][2]
                        elif  item_new_list[i][1][0]==item_new_list[j][2][0]:
                            item_new_list[i][1] = item_new_list[j][1]
                        elif  item_new_list[i][2][0]==item_new_list[j][1][0]:
                            item_new_list[i][2] = item_new_list[j][2]
                        elif  item_new_list[i][2][0]==item_new_list[j][2][0]:
                            item_new_list[i][2] = item_new_list[j][1]
                item_merge_list.append(item_new_list[i])
        # generate logic forms
        for i in range(len(item_merge_list)-1):
            for j in range(i+1, len(item_merge_list)):
                try:
                    logic_form_each = "Equals("
                    point1, crosspoint, point2 = id2name_dict[item_merge_list[i][1][1]], \
                                                                id2name_dict[item_merge_list[i][0]], \
                                                                    id2name_dict[item_merge_list[i][2][1]]
                    logic_form_each+='MeasureOf(Angle('+point1+", "+crosspoint+", "+point2+")), "
                    point1, crosspoint, point2 = id2name_dict[item_merge_list[j][1][1]], \
                                                                id2name_dict[item_merge_list[j][0]], \
                                                                    id2name_dict[item_merge_list[j][2][1]]
                    logic_form_each+='MeasureOf(Angle('+point1+", "+crosspoint+", "+point2+")))"
                except:
                    print(img_name,"error in sym angle!")
                    continue
                list_angle.append(logic_form_each)

    list_angle = list()
    angle_list, double_angle_list, triple_angle_list =[], [], []
    quad_angle_list, penta_angle_list = [], []

    for item in symangle:
        if not (len(item[1])==3 and item[1][0][0]=='p' and item[1][1][0]=='l' and item[1][2][0]=='l'):
            print(img_name+" relation of angle symbol does not conform to standard composition !")
            continue              
        if elem_dict[item[0]]['sym_class']=='angle':
            angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='double angle':
            double_angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='triple angle':
            triple_angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='quad angle':
            quad_angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='penta angle':
            penta_angle_list.append(item)

    if len(angle_list)>0: get_logic_form(angle_list, list_angle)
    if len(double_angle_list)>0: get_logic_form(double_angle_list, list_angle)
    if len(triple_angle_list)>0: get_logic_form(triple_angle_list, list_angle)
    if len(quad_angle_list)>0: get_logic_form(quad_angle_list, list_angle)
    if len(penta_angle_list)>0: get_logic_form(penta_angle_list, list_angle)

    return list_angle

def get_Textangle(id2name_dict, textangle, text_class_name, elem_dict, line_dict, sym2head_dict, img_name):

    list_angle = list()

    for item in textangle:
        sym = item[0]
        if len(item[1])!=3 or not ((item[1][0][0]=='p' and item[1][1][0]=='l' and item[1][2][0]=='l') or \
            (item[1][0][0]=='p' and item[1][1][0]=='p' and item[1][2][0]=='c')):
            print(img_name+" relation of angle/degree text does not conform to standard composition !")
            continue

        logic_form_each = "Equals("
        try:
            if item[1][2][0]=='l':
                if text_class_name=='angle':
                    point1, point2 = get_angle_point(item, elem_dict, line_dict, img_name, 'angle', elem_dict[sym]['text_content']) 
                    if point1==None or point2==None or elem_dict[sym]['text_content']==None:
                        print(img_name+" no corresponding angle for angle/degree text !")
                        continue
                    point1, crosspoint, point2 = id2name_dict[point1[0]], \
                                                    id2name_dict[item[1][0]], \
                                                        id2name_dict[point2[0]]
                    logic_form_each+='MeasureOf(Angle('+point1+", "+crosspoint+", "+point2+")), MeasureOf(angle "+elem_dict[sym]['text_content']+"))"
                    # logic_form_each+='MeasureOf(Angle('+point1+", "+crosspoint+", "+point2+")), MeasureOf(Angle("+elem_dict[sym]['text_content']+")))"
                elif text_class_name=='degree':
                    point1, point2 = get_angle_point(item, elem_dict, line_dict, img_name, 'degree', elem_dict[sym]['text_content']) 
                    if point1==None or point2==None or elem_dict[sym]['text_content']==None:
                        print(img_name+" no corresponding angle for angle/degree text !")
                        continue
                    point1, crosspoint, point2 = id2name_dict[point1[0]], \
                                                    id2name_dict[item[1][0]], \
                                                        id2name_dict[point2[0]]
                    logic_form_each+='MeasureOf(Angle('+point1+", "+crosspoint+", "+point2+")), "+elem_dict[sym]['text_content']+")"

            elif item[1][2][0]=='c':
                point1, point2, center = id2name_dict[item[1][0]], \
                                            id2name_dict[item[1][1]], \
                                                id2name_dict[item[1][2]]                           
                # logic_form_each+="MeasureOf(Angle("+point1+", "+center+", "+point2+")), "+elem_dict[sym]['text_content']+")"
                logic_form_each+='MeasureOf(Arc('+point1+", "+point2+")), "+elem_dict[sym]['text_content']+")"
        except:
            print(img_name, 'error in text angle/degree!')
            continue
        list_angle.append(logic_form_each)

    return list_angle

def get_logic_form(relation_each, elem_dict, image_id):

    logic_item = dict()
    id2name_dict = dict()
    line_dict = dict()  # line id -> [point id, point loc]
    circle_dict = dict()  # circle id -> [point id, point loc]
    sym2head_dict = dict() # sym(head) id -> head id 
    head2sym_dict = dict() # head id -> sym(head) id
    head_len_dict = dict() 

    points = relation_each['geos']['points']
    lines = relation_each['geos']['lines']
    circles = relation_each['geos']['circles']
    symbols = relation_each['symbols']

    textpoint2point_each, point2circle_each, point2line_each = [], [], []
    parallel_each, perpendicular_each, Bar_each = [], [], []
    textlen_each, symangle_each = [], []
    textangle_each, degree_each = [], []

    for item in relation_each['relations']["geo2geo"]:
        if item[1][0]=='c':
            point2circle_each.append(item)
        elif item[1][0]=='l':
            point2line_each.append(item)

    for item in relation_each['relations']["sym2sym"]:
        sym2head_dict[item[0]]=item[1] # list 
        if len(item[1])==2:
            head2sym_dict[item[1][0]]=item[0] # scale
            head2sym_dict[item[1][1]]=item[0] 
        else:
            head2sym_dict[item[1][0]]=item[0] 

    for item in relation_each['relations']["sym2geo"]:
        if 'parallel' in elem_dict[item[0]]['sym_class']: parallel_each.append(item)
        if elem_dict[item[0]]['text_class']=='point': textpoint2point_each.append(item)
        if elem_dict[item[0]]['sym_class']=='perpendicular': perpendicular_each.append(item)
        if 'bar' in elem_dict[item[0]]['sym_class']: Bar_each.append(item)
        if elem_dict[item[0]]['text_class']=='len': textlen_each.append(item)
        if 'angle' in elem_dict[item[0]]['sym_class']: symangle_each.append(item)
        if elem_dict[item[0]]['text_class']=='angle': textangle_each.append(item)
        if elem_dict[item[0]]['text_class']=='degree': degree_each.append(item)

        if elem_dict[item[0]]['sym_class']=='head_len': 
            head_len_id = elem_dict[item[0]]['id']
            sym_id = head2sym_dict[head_len_id]
            if elem_dict[sym_id]['sym_class']=='head':
                sym_id = head2sym_dict[sym_id]
            if not sym_id in head_len_dict:
                head_len_dict[sym_id] = []
            head_len_dict[sym_id] += item[1]
    
        if elem_dict[item[0]]['sym_class']=='head':
            head_id = item[0]
            sym_id = head2sym_dict[head_id]
            elem_dict[head_id]['text_content'] = elem_dict[sym_id]['text_content']
            if elem_dict[sym_id]['text_class']=='point': textpoint2point_each.append(item)
            if elem_dict[sym_id]['text_class']=='len': textlen_each.append(item)
            if elem_dict[sym_id]['text_class']=='angle': textangle_each.append(item)
            if elem_dict[sym_id]['text_class']=='degree': degree_each.append(item)
    
    for key, value in head_len_dict.items():
        textlen_each.append([key, value])

    point_instances, point_positions, circle_instances = \
                get_point_name(points, circles, textpoint2point_each, point2circle_each, elem_dict, id2name_dict)

    line_instances = get_line_instances(lines, point2line_each, elem_dict, id2name_dict, line_dict)
    # PointLiesOnLine
    list_PointLiesOnLine =  get_PointLiesOnLine(id2name_dict, line_dict, image_id)
    # PointLiesOnCircle
    list_PointLiesOnCircle = get_PointLiesOnCircle(elem_dict, id2name_dict, point2circle_each, circles, circle_dict, image_id)
    # Parallel
    list_Parallel = get_Parallel(id2name_dict, line_dict, parallel_each, elem_dict, image_id)
    # Perpendicular
    list_Perpendicular = get_Perpendicular(id2name_dict, line_dict, perpendicular_each, elem_dict, image_id)
    # Bar
    list_Bar = get_Bar(id2name_dict, Bar_each, elem_dict, image_id)
    # Text_len
    list_Textlen = get_Textlen(id2name_dict, textlen_each, elem_dict, sym2head_dict, circles, image_id)
    # Sym_angle
    list_Symangle = get_Symangle(id2name_dict, symangle_each, elem_dict, line_dict, image_id)
    # Text_angle
    list_Textangle = get_Textangle(id2name_dict, textangle_each, 'angle', elem_dict, line_dict, sym2head_dict, image_id)
    # degree
    list_Degree = get_Textangle(id2name_dict, degree_each, 'degree', elem_dict, line_dict, sym2head_dict, image_id)

    diagram_logic_forms = list_PointLiesOnLine+list_PointLiesOnCircle+list_Parallel+list_Perpendicular \
                            +list_Bar+list_Textlen+list_Symangle+list_Textangle+list_Degree
    # remove the same logic forms
    rm_dup_logic_forms = list(set(diagram_logic_forms))
    rm_dup_logic_forms.sort(key=diagram_logic_forms.index)


    logic_item['point_instances']=point_instances
    logic_item["line_instances"] = line_instances
    logic_item['circle_instances']=circle_instances
    logic_item["diagram_logic_forms"] = rm_dup_logic_forms
    logic_item['point_positions']=point_positions
    
    return logic_item, id2name_dict
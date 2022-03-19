

import json
import metric
import argparse
import numpy as np
from itertools import product, combinations
import re
import os

import sys
sys.path.append("../../symbolic_solver")

from extended_definition import ExtendedDefinition
from logic_parser import LogicParser


def match_points(A, B):
    # Give two point sets A and B, return the correspondence of their points.
    thresh_min_dis = 15
    dis = lambda x, y: ((x[0]-y[0])**2 + (x[1]-y[1])**2) ** 0.5
    AtoB = {x: None for x in A.keys()}
    BtoA = {x: None for x in B.keys()}
    mismatchA, mismatchB = set(A.keys()), set(B.keys())
    while len(mismatchA) > 0 and len(mismatchB) > 0:
        # consider the situation that with different keys when having explicit point key 
        # penalty factor is 5.0
        f = lambda p, q: dis(A[p], B[q]) + float(p != q) * 5.0
        mindis, p, q = min([(f(p, q), p, q) for p in mismatchA for q in mismatchB])
        if mindis > thresh_min_dis: break
        mismatchA.remove(p)
        mismatchB.remove(q)
        AtoB[p] = q
        BtoA[q] = p

    return AtoB, BtoA

def generate_GeoSolver(point_positions, lines, circles, diagram_logic_forms):
    '''
        Start the GeoSolver Engine to help check the accuracy of 'Perpendicular'. 
    '''

    isLetter = lambda ch: ch.isalpha() and len(ch) == 1
    parser = LogicParser(ExtendedDefinition(False))

    parser.logic.point_positions = point_positions
    parser.logic.define_point([p for p in parser.logic.point_positions if isLetter(p)])

    # consider same center circle
    for point in circles:
        parser.logic.define_circle(point[0])

    for line in lines:
        if len(line) == 2 and isLetter(line[0]) and isLetter(line[1]):
            parser.logic.define_line(line[0], line[1])

    diagram_logic_forms = sorted(diagram_logic_forms, key=lambda x: x[0] == "Perpendicular")

    for logic_form in diagram_logic_forms:
        try:
            parser.dfsParseTree(logic_form)
        except Exception as e:
            #print("\033[0;0;41mError:\033[0m", repr(e))
            print(repr(e))
    return parser

def strip(lst):
    return [x for x in lst if str(x).strip() != ""]

def div(a, b):
    if b == 0:
        return 1.0
    return a * 1.0 / b

def calc_f1(acc, recall):
    return 2.0 * acc * recall / max(acc + recall, 1e-6)

def ok(forms):
    if type(forms) == list:
        return all([ok(x) for x in forms])
    return forms.find('point_') == -1

def diagram_evaluaion(graph_gt, graph_test):
    '''
    Paras: two results of diagram parsing (typically ground truth & test)
    Returns: Accuracy, Recall, IoU
    '''
    Accuracy = metric.DiagramResult()
    Recall = metric.DiagramResult()
    IoU = metric.DiagramResult()
    if graph_gt == None or graph_test == None:
        return Accuracy, Recall, IoU

    # Do some preparation
    graphs = [graph_gt, graph_test]
    point_positions = [x["point_positions"] for x in graphs]
    lines = [strip(x["line_instances"]) for x in graphs]
    circles = [strip(x["circle_instances"]) for x in graphs]
    diagram_logic_forms = [metric.parse_logic_forms(strip(x["diagram_logic_forms"])) for x in graphs]
    diagram_logic_forms_geo2geo = [[],[]]
    diagram_logic_forms_geo2sym = [[],[]]
    
    # Some points may be represented as 'point_10' rather than 'A'(Inter-GPS), so we just erase them.
    for i in range(2):
        point_positions[i] = {x:y for x, y in point_positions[i].items() if (type(x) == str) and (not 'point'in x)}
        lines[i] = [x for x in lines[i] if not 'point' in x]
        # consider same center circle
        circles[i] = [x for x in circles[i] if not 'point' in x]
        diagram_logic_forms[i] = [x for x in diagram_logic_forms[i] if ok(x)]
        for item in diagram_logic_forms[i]:
            if item[0] in ['PointLiesOnLine', 'PointLiesOnCircle']: 
                diagram_logic_forms_geo2geo[i].append(item)
            else:
                diagram_logic_forms_geo2sym[i].append(item)
        assert len(diagram_logic_forms_geo2geo[i])+len(diagram_logic_forms_geo2sym[i])==len(diagram_logic_forms[i])

    # Find the correspondence of points.
    mp = match_points(*point_positions)

    for i in range(2):

        j = i^1
        result = Recall if i == 0 else Accuracy
        result['logic_forms_all'] = 0
        result['logic_forms_geo2geo'] = 0
        result['logic_forms_geo2sym'] = 0

        parser = generate_GeoSolver(point_positions[i], lines[i], circles[i], diagram_logic_forms[i])

        for logic_form in diagram_logic_forms[i]:

            # generate all possible logic form without further inference 
            all_possibles = metric.generate_all(logic_form, parser, point_positions[i])
            
            if logic_form[0] == "Perpendicular" or logic_form[0] == "Parallel" :
                A = parser.logic.find_all_points_on_line(logic_form[1][1:])
                B = parser.logic.find_all_points_on_line(logic_form[2][1:])
                all_possibles = []
                for a, b in combinations(A, 2):
                    for c, d in combinations(B, 2):
                        logic_form[1][1:] = [a, b]
                        logic_form[2][1:] = [c, d]
                        all_possibles.extend(metric.generate_all(logic_form, parser, point_positions[i]))

            if any([metric.same(p, q, mp[i]) for p, q in product(all_possibles, diagram_logic_forms[j])]):
                result['logic_forms_all'] += 1
                if logic_form[0] in ['PointLiesOnLine', 'PointLiesOnCircle']:
                    result['logic_forms_geo2geo'] += 1
                else:
                    result['logic_forms_geo2sym'] += 1

        result['points'] = sum([x != None for x in mp[i].values()])

        # 寻找正确的点的名称（有些点的名称不是大写字母）
        for s in lines[i]:
            if len(s)==2: 
                s0, s1 = s[0], s[1]
            else:
                s0, s1 = None, None
                for index in range(len(s)-1):
                    if s[0:index+1] in mp[i] and s[index+1:] in mp[i]:
                        s0, s1 = s[0:index+1], s[index+1:]
                        break                    
            if (str(mp[i][s0])+str(mp[i][s1])) in lines[j] or (str(mp[i][s1])+str(mp[i][s0])) in lines[j]:
                result['lines'] +=1
        result['circles'] = sum([mp[i].get(x[0], None) != None for x in circles[i]]) # consider same center circle

        IoU['logic_forms_all'] = div(result['logic_forms_all'], len(diagram_logic_forms[0]) + len(diagram_logic_forms[1]) - result['logic_forms_all'])
        IoU['logic_forms_geo2geo'] = div(result['logic_forms_geo2geo'], len(diagram_logic_forms_geo2geo[0]) + len(diagram_logic_forms_geo2geo[1]) - result['logic_forms_geo2geo'])
        IoU['logic_forms_geo2sym'] = div(result['logic_forms_geo2sym'], len(diagram_logic_forms_geo2sym[0]) + len(diagram_logic_forms_geo2sym[1]) - result['logic_forms_geo2sym'])
        IoU['points'] = div(result['points'], len(point_positions[0]) + len(point_positions[1]) - result['points'])
        IoU['lines'] = div(result['lines'], len(lines[0]) + len(lines[1]) - result['lines'])
        IoU['circles'] = div(result['circles'], len(circles[0]) + len(circles[1]) - result['circles'])
        
        result['logic_forms_all'] = div(result['logic_forms_all'], len(diagram_logic_forms[i]))
        result['logic_forms_geo2geo'] = div(result['logic_forms_geo2geo'], len(diagram_logic_forms_geo2geo[i]))
        result['logic_forms_geo2sym'] = div(result['logic_forms_geo2sym'], len(diagram_logic_forms_geo2sym[i]))
        result['points'] = div(result['points'], len(list(mp[i].values())))
        result['lines'] = div(result['lines'], len(lines[i]))
        result['circles'] = div(result['circles'], len(circles[i]))
    
    return Accuracy, Recall, IoU

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--test_set_path', default="/lustre/home/mlzhang/Datasets/PGDP5K/test")
    parser.add_argument('--diagram_gt', default="/lustre/home/mlzhang/Datasets/PGDP5K/our_diagram_logic_forms_annot.json")
    parser.add_argument('--diagram_pred', default="/lustre/home/mlzhang/GeoMathQA/Geo_parsing_GAT/inference/PGDP5K_geo_MNV2_FPN_2x_VIS_SEM_LOC_large_24/model_final.pth_True_1_0.5/logic_forms_pred.json")
    
    parser = parser.parse_args()

    with open(parser.diagram_gt, "r") as f1:
        gt = json.load(f1)
    with open(parser.diagram_pred, "r") as f2:
        pred = json.load(f2)
    
    AccuracyList = []
    RecallList = []
    IoUList = []

    # img_list = [item.replace('.png','') for item in os.listdir(parser.test_set_path)]
    for key in pred:
        # if key!='Screenshot - 2021-11-04 16.17.54_600.png':
        print(key)
        Accuracy, Recall, IoU = diagram_evaluaion(gt.get(key, None), pred.get(key, None))
        AccuracyList.append(Accuracy)
        RecallList.append(Recall)
        IoUList.append(IoU)  

    for element in ['points', 'lines', 'circles', 'logic_forms_all','logic_forms_geo2geo','logic_forms_geo2sym']:
        accuracy = np.mean([x[element] for x in AccuracyList])
        # print([x[element] for x in AccuracyList])
        recall = np.mean([x[element] for x in RecallList])
        f1score = calc_f1(accuracy, recall)
        iou = np.mean([x[element] for x in IoUList])
        print("Average " + element + " result among test data:   Accuracy: %.2f%%  Recall: %.2f%%  F1 Score: %.2f%%  IoU: %.2f%%" % (accuracy*100, recall*100, f1score*100, iou*100))
    
    for element in ['logic_forms_all','logic_forms_geo2geo','logic_forms_geo2sym']:
        print('#################### '+element+' ########################')
        number = sum([calc_f1(x[element], y[element]) == 1.0 for x, y in zip(AccuracyList, RecallList)])
        print ("Totally Same (F1 Score = 100%%): %.2f%%" % (number / len(AccuracyList) * 100))
        number = sum([x[element] == 1.0 for x in RecallList])
        print ("Perfect Recall (Recall = 100%%): %.2f%%" % (number / len(AccuracyList) * 100))
        number = sum([calc_f1(x[element], y[element]) >= 0.75 for x, y in zip(AccuracyList, RecallList)])
        print ("Almost Same (F1 Score >= 75%%): %.2f%%" % (number / len(AccuracyList) * 100))
        number = sum([calc_f1(x[element], y[element]) >= 0.5 for x, y in zip(AccuracyList, RecallList)])
        print ("Likely Same (F1 Score >= 50%%): %.2f%%" % (number / len(AccuracyList) * 100))
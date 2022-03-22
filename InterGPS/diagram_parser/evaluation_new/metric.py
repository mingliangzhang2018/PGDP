from pyparsing import Optional, Word, alphanums, Forward, Group, Word, Literal, ZeroOrMore
from itertools import permutations
import math
import numpy as np


identifier = Word(alphanums + ' +-*/.\\\{\}^_$\'=')
lparen = Literal("(").suppress()
rparen = Literal(")").suppress()
expression = Forward()
arg = Group(expression) | identifier 
args = arg + ZeroOrMore( Literal(",").suppress() + arg)
expression <<= identifier + lparen + Optional(args)  + rparen


def DiagramResult():
    ret = {}
    ret['points'] = 0.0
    ret['lines'] = 0.0
    ret['circles'] = 0.0
    ret['logic_forms_all'] = 0.0
    ret['logic_forms_geo2geo'] = 0.0
    ret['logic_forms_geo2sym'] = 0.0
    return ret

def parse(tree):
    parseResult = expression.parseString(tree).asList()
    return parseResult

def parse_logic_forms(logic_forms):
    '''
        Give a piece of logic forms in string, generate the corresponding list form.
        The logic form after parsing: 
            ['Equals', ['LengthOf', ['Line', 'A', 'B']], '13']
            ['Perpendicular', ['Line', 'B', 'D'], ['Line', 'D', 'C']] 
            ...
    '''
    ret = []
    for logic_form in logic_forms:
        try:
            now = parse(logic_form.replace('|','').replace(' ', ''))
            ret.append(now)
        except Exception as e:
            print("\033[0;0;41mError:\033[0m", repr(e))
    return ret

def listmul(group, lst):
    '''
    Give a group of list1, list2, list3, ... and a list of element1, element2, element3, ...
    Return [list1 + element1, list1 + element2, ..., list2 + element1, list2 + element2, ..., ...]
    '''
    ret = []
    for g in group:
        for l in lst:
            ret.append(g + [l])
    return ret

def generate_all(phrase, parser, point_positions):
    ''' 
    Give a piece of logic form, return a list contains all its equivalent forms.
    e.g. 
        phrase = ['Equals', ['LengthOf', ['Line', 'A', 'B']], 2]
        return: [['Equals', ['LengthOf', ['Line', 'A', 'B']], 2],
                 ['Equals', ['LengthOf', ['Line', 'B', 'A']], 2],
                 ['Equals', 2, ['LengthOf', ['Line', 'B', 'A']]],
                 ['Equals', 2, ['LengthOf', ['Line', 'A', 'B']]]]
    '''
    if type(phrase) != list:
        return [phrase]
    
    content = [generate_all(element, parser, point_positions) for element in phrase[1:]]

    # Can not change
    if len(phrase) == 2 or phrase[0] in ["Circle", "PointLiesOnLine", "PointLiesOnCircle", "BisectsAngle", \
                                        "InscribedIn", "IsMidpointOf", "LegOf", "IsCentroidOf", "IsIncenterOf", \
                                        "IsRadiusOf", "IsDiameterOf", "IsMidsegmentOf", "IsChordOf", "IsDiagonalOf", \
                                        "RatioOf"]:
        ret = [[phrase[0]]]
        for x in content:
            ret = listmul(ret, x)
        return ret
    # Any order
    elif phrase[0] == "Arc" and len(phrase) == 3 or phrase[0] in ["Equals", "Line", "Parallel", "Perpendicular", \
                                                                 "Congruent", "Similar", "Add", "Mul", "SumOf"]:
        ret = []
        for cur in permutations(content):
            expr = [[phrase[0]]]
            for x in cur:
                expr = listmul(expr, x)
            ret.extend(expr)
        return ret
    # Any order but not the last
    elif phrase == "IntersectAt":
        ret = []
        for cur in permutations(content[0:-1]):
            expr = [[phrase[0]]]
            for x in cur:
                expr = listmul(expr, x)
            expr = listmul(expr, content[-1])
        ret.extend(expr)
        return ret
    # The start and end positions can be swapped
    elif phrase[0] == "Arc" and len(phrase) == 4:
        ret1 = [[phrase[0]]]
        for x in content:
            ret1 = listmul(ret1, x)
        content[0], content[-1] = content[-1], content[0]
        ret2 = [[phrase[0]]]
        for x in content:
            ret2 = listmul(ret2, x)
        return ret1 + ret2
    # Find same direction of point on the line
    elif phrase[0] in ["Angle"]:
        cross_point = content[1][0][0]
        cross_point_loc=point_positions[cross_point]
        point_selct_list =[[],[]]
        for index in range(2):
            point_edge = content[index*2][0][0]
            point_edge_loc = point_positions[point_edge]
            point_list = parser.logic.find_all_points_on_line([point_edge, cross_point])
            for point in point_list:
                if point!=cross_point:
                    point_loc = point_positions[point]
                    if (point_loc[0]-cross_point_loc[0])*(point_edge_loc[0]-cross_point_loc[0]) + \
                        (point_loc[1]-cross_point_loc[1])*(point_edge_loc[1]-cross_point_loc[1])>0:
                        point_selct_list[index].append(point)
        ret1 = [[phrase[0]]]
        ret1 = listmul(ret1, point_selct_list[0])
        ret1 = listmul(ret1, [cross_point])
        ret1 = listmul(ret1, point_selct_list[1])
        ret2 = [[phrase[0]]]
        ret2 = listmul(ret2, point_selct_list[1])
        ret2 = listmul(ret2, [cross_point])
        ret2 = listmul(ret2, point_selct_list[0])
        return ret1 + ret2
    # Cycle
    elif phrase[0] in ["Triangle", "Polygon", "Quadrilateral", "Parallelogram", 
                       "Rhombus", "Rectangle", "Square", "Trapezoid", "Kite"]:
        ret = []
        for i in range(len(content)):
            expr = []
            for x in content[i:] + content[:i]:
                expr = listmul(expr, x)
            ret.extend(expr)
        return ret
    else:
        print ("Not found", phrase)
        ret = [[phrase[0]]]
        for x in content:
            ret = listmul(ret, x)
        return ret

def same(phrase1, phrase2, mp):

    '''
    Check whether two logic forms are same.
    Note that all the points in phrase1 should be replaced by dict mp.
    '''
    if type(phrase1) != type(phrase2):
        return False

    if type(phrase1) != list:
        if phrase1.find("radius_") != -1 and phrase2.find("radius_") != -1:
            return True
        #if len(phrase1) == 1 and phrase1[0] >= 'A' and phrase1[0] <= 'Z':
        phrase1 = mp.get(phrase1, phrase1)
        return phrase1 == phrase2

    if len(phrase1) != len(phrase2):
        return False

    return all([same(p, q, mp) for p, q in zip(phrase1, phrase2)])


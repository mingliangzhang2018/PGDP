import os
import json
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import cv2

from geo_parse.structures.bounding_box import BoxList
from geo_parse.structures.geo_ins import GeoList
from geo_parse.structures.rel_map import RelList

class GEODataset(torch.utils.data.Dataset):
    
    CLASSES_SYM = [
        "__background__", 
        "text", 
        "perpendicular", "head", "head_len",
        "angle","bar","parallel", 
        "double angle","double bar","double parallel", 
        "triple angle","triple bar","triple parallel",
        "quad angle", "quad bar", 
        "penta angle", 
    ]
    CLASSES_GEO = [
        "__background__", 
        "point","line","circle"
    ]
    CLASSES_TEXT = [
        "point", "line", "len", "angle", "degree", "area", 
    ]
    CLASSES_SEM = [
        "point","line","circle",
        "text", 
        "perpendicular", "head", "head_len",
        "angle", "bar", "parallel"
    ]
    CLASSES_REL = [
       "__background__", 
        "geo2geo", "text2geo", "sym2geo", "text2head"
    ]
    
    THICKNESS = 3

    def __init__(self, root, ann_file, transforms=None, is_train=False, cfg=None): 
        
        self.img_root = root
        self.transforms = transforms
        self.ids = []
        self.is_train = is_train
        self.cfg = cfg
        self.class_sym_to_ind = dict(zip(GEODataset.CLASSES_SYM, range(len(GEODataset.CLASSES_SYM))))
        self.class_geo_to_ind = dict(zip(GEODataset.CLASSES_GEO, range(len(GEODataset.CLASSES_GEO))))
        self.class_text_to_ind = dict(zip(GEODataset.CLASSES_TEXT, range(len(GEODataset.CLASSES_TEXT))))

        with open(ann_file, 'rb') as file:
            self.contents = json.load(file)
        for key in self.contents.keys():
            self.ids.append(key)

    def __getitem__(self, index):
        
        img_id = self.ids[index]
        annot_each = self.contents[img_id]

        # Samples without non-geometric primitives do not participate in the training
        while self.is_train and len(annot_each['symbols'])==0:
            index = np.random.randint(0, len(self.ids))
            img_id = self.ids[index]
            annot_each = self.contents[img_id]

        img = Image.open(os.path.join(self.img_root, annot_each['file_name'])).convert("RGB")

        target_det = self.get_target_det(annot_each, img.size)
        target_seg = self.get_target_seg(annot_each, img.size)
        target_rel = self.get_target_rel(annot_each, target_det, target_seg)

        if self.transforms is not None:
            img, target_det, target_seg = self.transforms(img, target_det, target_seg)

        return img, target_det, target_seg, target_rel, index

    def __len__(self):
        return len(self.ids)

    def get_DET_GT_each(self, symbols):
        boxes, classes, text_contents, ids, classes_text = [], [], [], [], []
        # arrow and others primitives are excluded 
        for obj in symbols:
            if obj["sym_class"]!='text' and obj["sym_class"]!='arrow':
                classes.append(self.class_sym_to_ind[obj["sym_class"]])
                classes_text.append(-1)
                boxes.append(obj["bbox"])
                text_contents.append(obj["text_content"])
                ids.append(obj["id"])
            elif obj["text_class"]!='others':
                classes.append(self.class_sym_to_ind['text'])
                classes_text.append(self.class_text_to_ind[obj["text_class"]])
                boxes.append(obj["bbox"])
                text_contents.append(obj["text_content"])
                ids.append(obj["id"])

        return boxes, classes, text_contents, ids, classes_text
    
    def get_SEG_GT_each(self, geos, img_size):
        masks, locs, classes, ids = [], [], [], []
        width, height = img_size

        for key, values in geos.items():
            for item in values:
                mask = np.zeros((height,width), dtype=np.uint8)
                loc = self.get_round_loc(item['loc'])
                if key=='points':
                    cv2.circle(mask, center=loc[0], radius=GEODataset.THICKNESS, color=255, thickness=-1)
                if key=='lines':
                    cv2.line(mask, pt1=loc[0], pt2=loc[1], color=255, thickness=GEODataset.THICKNESS)
                if key=='circles':
                    cv2.circle(mask, center=loc[0], radius=loc[1], color=255, thickness=GEODataset.THICKNESS)
                    if loc[2]!='1111': # consider the case of a nonholonomic circle
                        draw_pixel_loc = np.where(mask>0)
                        for x, y in zip(draw_pixel_loc[1],draw_pixel_loc[0]):
                            delta_x = x-loc[0][0] # + - - +
                            delta_y = y-loc[0][1] # - - + +
                            if delta_x>=0 and delta_y<=0 and loc[2][0]=='0': mask[y,x] = 0
                            if delta_x<=0 and delta_y<=0 and loc[2][1]=='0': mask[y,x] = 0
                            if delta_x<=0 and delta_y>=0 and loc[2][2]=='0': mask[y,x] = 0
                            if delta_x>=0 and delta_y>=0 and loc[2][3]=='0': mask[y,x] = 0

                masks.append(mask)
                classes.append(self.class_geo_to_ind[key[:-1]])
                locs.append(item['loc'])
                ids.append(item['id'])
        return np.stack(masks, axis=-1), locs, classes, ids

    def get_round_loc(self, loc):
        loc_copy = []
        for item in loc:
            if isinstance(item, float):
                item = round(item)
            if isinstance(item, list):
                item = tuple([round(v) for v in item])
            loc_copy.append(item)
        return loc_copy 

    def get_img_info(self, index):
        img_id = self.ids[index]
        annot_each = self.contents[img_id]
        return {"height": annot_each['height'], "width": annot_each['width']}

    def get_groundtruth(self, index):

        img_id = self.ids[index]
        annot_each = self.contents[img_id]
        height, width = annot_each['height'], annot_each['width']
        target_det = self.get_target_det(annot_each, (width, height))
        target_seg = self.get_target_seg(annot_each, (width, height))
        target_rel = self.get_target_rel(annot_each, target_det, target_seg)
        return target_det, target_seg, target_rel

    def get_target_det(self, annot_each, img_size):

        boxes, classes, text_contents, ids, classes_text = self.get_DET_GT_each(annot_each['symbols'])
        boxes = torch.as_tensor(boxes).reshape(-1, 4) 
        target_det = BoxList(boxes, img_size, mode="xywh").convert("xyxy")
        target_det.add_field("labels", torch.tensor(classes))
        target_det.add_field("text_contents", text_contents)
        target_det.add_field("ids", ids)
        target_det.add_field("labels_text", classes_text)
        return  target_det

    def get_target_seg(self, annot_each, img_size):

        masks, locs, classes, ids = self.get_SEG_GT_each(annot_each['geos'], img_size)
        target_seg = GeoList(masks, img_size)
        target_seg.add_field("labels", classes)
        target_seg.add_field("locs", locs)
        target_seg.add_field("ids", ids)
        return target_seg

    def get_target_rel(self, annot_each, target_det, target_seg):
        ids = target_det.get_field('ids') + target_seg.get_field('ids')
        labels_all = target_det.get_field('labels').tolist() + target_seg.get_field('labels')
        labels_text = target_det.get_field('labels_text') + [-1]*len(target_seg.get_field('labels'))
        target_rel = RelList(ids, labels_all, labels_text)
        target_rel.construct_edge_label(annot_each['relations'])
        return target_rel
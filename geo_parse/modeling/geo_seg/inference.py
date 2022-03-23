import numpy as np
from sklearn.cluster import MeanShift
import torch
from skimage import measure
from geo_parse.structures.geo_ins import GeoList
import cv2
from scipy import optimize
import math


class GEOPostProcessor(torch.nn.Module):
    """
        Performs post-processing on the outputs of the geo masks.
        This is only used in the testing.
    """
    def __init__(self, cfg):

        super(GEOPostProcessor, self).__init__()
        self.cfg = cfg
        self.band_width = cfg.MODEL.SEG.LOSS_DELTA_D/2.2
        self.min_area_ratio = cfg.MODEL.SEG.MIN_AREA_RATIO
        self.bin_seg_th = cfg.MODEL.SEG.BIN_SEG_TH
        
    def forward(self, bin_seg, embedding, image_sizes, fpn_stride):
        """
        Arguments:
            binary_seg: N*3*H*W [Tensor]
            embedding: N*EMB_DIMS*H*W [Tensor]
            image_sizes: 
        Returns:
            geolists (list[GeoList]): the post-processed masks, after
                applying mean-shift cluster and other methods
        """
        geolists = []

        for bin_seg_i, embedding_i, img_size in zip(bin_seg, embedding, image_sizes):
            embedding_i = embedding_i.permute(1,2,0).cpu().numpy()
            bin_seg_i = torch.sigmoid(bin_seg_i).permute(1,2,0).cpu().numpy()
            bin_seg_i = (bin_seg_i>self.bin_seg_th).astype(np.uint8)*255
            img_size = (round(img_size[1]/fpn_stride), round(img_size[0]/fpn_stride))
            geolist = self.forward_for_single_img(bin_seg_i, embedding_i, img_size)
            geolists.append(geolist)

        return geolists


    def forward_for_single_img(self, bin_seg, embedding, img_size):
        """
        First use mean shift to find dense cluster center.
        Arguments:
            bin_seg: numpy [H, W, 3], each pixel is 0 or 1, 0 is background pixel
            embedding: numpy [H, W, embed_dim]
            band_width: cluster pixels of the same instance within the range of band_width from the center, 
                        less small band_width is better for distinguishing similar instances  
        Return:
            cluster_result: numpy [H, W], index of different instances on each pixel
        """

        masks, classes = [], []
        width, height = img_size[0], img_size[1]

        # get point instance
        label_img = measure.label(bin_seg[:,:,0], connectivity = 2)
        avg_area_point = (label_img>0).sum()/label_img.max()
        for idx in np.unique(label_img):
            if idx!=0:
                mask = (label_img==idx)[:height, :width]
                if mask.sum() > avg_area_point*self.min_area_ratio[0]:
                    masks.append(mask)
                    classes.append(1)
        
        # get line and circle instance 
        bin_seg_lc = np.sum(bin_seg[:,:,1:], axis=-1)
        cluster_result = np.zeros(bin_seg_lc.shape, dtype=np.int32)
        cluster_list = embedding[bin_seg_lc>0]
        if len(cluster_list)==0:
            return self.construct_GeoList(masks, classes, img_size)
        # cluster pixels into instances
        mean_shift = MeanShift(bandwidth=self.band_width, bin_seeding=True)
        mean_shift.fit(cluster_list)
        labels = mean_shift.labels_
        cluster_result[bin_seg_lc>0] = labels + 1
        avg_area_lc = (cluster_result>0).sum()/cluster_result.max()

        # screen out line, circle and noise instance
        for idx in np.unique(cluster_result):
            if idx!=0:
                mask = (cluster_result==idx)[:height, :width]
                line_pixels = bin_seg[:height, :width, 1][mask].sum()
                circle_pixels = bin_seg[:height, :width ,2][mask].sum()
                # filter out noise
                label_img = measure.label(mask, connectivity = 2)
                props = measure.regionprops(label_img)
                max_area = 0
                for prop in props:
                    max_area = max(max_area, prop.area)
                    if prop.area<avg_area_lc*0.01: 
                        mask[label_img==prop.label]=False
                # classify geo instances (line and circle)  
                if line_pixels>circle_pixels and max_area>avg_area_lc*self.min_area_ratio[1]:
                    # masks.append(mask)
                    # classes.append(2)
                    self.seperate_ins(mask, 2, masks, classes, avg_area_lc)
                if line_pixels<=circle_pixels and max_area>avg_area_lc*self.min_area_ratio[2]:
                    masks.append(mask)
                    classes.append(3)
                    # self.seperate_ins(mask, 3, masks, classes, avg_area_lc)
        
        return self.construct_GeoList(masks, classes, img_size)

    def seperate_ins(self, mask, label, masks, classes, avg_area_lc):
        '''
            some case: diffenent similar instances are clustered into one instance 
            this function sperate them according to spatial location
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask_new = cv2.dilate(mask.astype(np.uint8)*255, kernel) 
        label_img = measure.label(mask_new, connectivity = 2)
        for idx in np.unique(label_img):
            if idx!=0:
                mask_ins = (label_img==idx)&mask
                if mask_ins.sum()>avg_area_lc*self.min_area_ratio[label-1]:
                    masks.append(mask_ins)
                    classes.append(label)

    def get_parse_loc(self, classes, masks, ids):
        """
            obtain the parsing location of geo instance
        """
        def f_2(center_loc, data_loc):
            """ 
            Arguments:
                data_loc: ndarray[num, 2]
                center_loc: ndarray[2]
            Returns:
                fitting error
            """
            Ri = np.linalg.norm(data_loc-center_loc,axis=1)
            return Ri - Ri.mean()

        def get_point_dist(p0, p1):
            dist = math.sqrt((p0[0]-p1[0])**2
                    +(p0[1]-p1[1])**2)
            return dist

        locs = []
        point_locs = {}
        point_name_list = []
        
        for class_index, mask, id in zip(classes, masks, ids):
            if class_index==1: # point
                y_list, x_list = np.where(mask>0) 
                point_locs[id]=[x_list.mean(), y_list.mean()] # center of the pixel set
                point_name_list.append(id)

        for class_index, mask, id in zip(classes, masks, ids):
            if class_index==1: # point
                locs.append([point_locs[id]])
            elif class_index==2: # line
                mask_loc = np.where(mask>0)
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
                line_loc = []
                # the endpoints of line are determined by the point instances
                for endpoint_loc in [endpoint_loc_1, endpoint_loc_2]:
                    point_name_list.sort(key=lambda x: get_point_dist(point_locs[x], endpoint_loc))
                    if point_name_list[0]!=nearst_point: 
                        nearst_point = point_name_list[0]
                    else:
                        nearst_point = point_name_list[1]
                    line_loc.append(point_locs[nearst_point])
                locs.append(line_loc)
            elif class_index==3: # circle
                point_list = np.stack(list(np.where(mask>0)),axis=-1)
                center_estimate = np.mean(point_list, axis=0) 
                # The center location of circles are obtained by the least square method
                center, _ = optimize.leastsq(f_2, center_estimate, args=(point_list))
                radius = np.linalg.norm(point_list-center,axis=1).mean()
                # TODO get explicit quadrant
                locs.append([[center[1], center[0]], radius, '1111'])
        
        return locs

    def construct_GeoList(self, masks, classes, size):
        
        masks_new = np.stack(masks, axis=-1).astype(np.uint8)*255
        ids = []
        id_num = [0, 0, 0]
        for item in classes:
            if item==1: 
                ids.append('p'+str(id_num[item-1]))
            elif item==2: 
                ids.append('l'+str(id_num[item-1]))
            elif item==3:
                ids.append('c'+str(id_num[item-1]))
            id_num[item-1] +=1

        pred_seg = GeoList(masks_new, size)
        pred_seg.add_field("labels", classes)
        pred_seg.add_field("locs", self.get_parse_loc(classes, masks, ids))
        pred_seg.add_field("ids", ids)

        return pred_seg

def make_seg_postprocessor(config):

    geo_selector = GEOPostProcessor(config)

    return geo_selector



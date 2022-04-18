import cv2
import numpy as np

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
TO_REMOVE = 1

class GeoList(object):
    
    """
        This class represents a set of geometric primitive instances.
        The geo instance are represented as
            1. masks
                N*M*num_inst unit8 numpy array map
            2. locs
                parse location
                'point': [[x,y]], # point location
                'line': [[x1,y1],[x2,y2]], # point1 location and point2 location
                'circle': [[x,y],r,'1111'], # center location, radius and 4 quadrant(0/1)
            3. labels
                {'__background__':0, 'point':1, 'line':2, 'circle':3}
        In order to uniquely determine the geo masks with respect
        to an image, we also store the corresponding image dimensions.
    """

    def __init__(self, masks, image_size):

        if masks.ndim!= 3:
            raise ValueError(
                "masks should have 3 dimensions, got {}".format(masks.ndim)
            )
        self.masks = masks
        self.size = image_size  # (image_width, image_height)
        self.extra_fields = {}
        
    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def resize(self, size):
        """
            Returns a resized copy of this masks and loc
            :param size: The requested size in pixels, as a 2-tuple (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_width, ratio_height = ratios
        self.masks = cv2.resize(self.masks, size, interpolation=cv2.INTER_NEAREST)
        if 'locs' in self.extra_fields.keys():
            loc_copy = []
            for loc in self.extra_fields['locs']:
                item_each = []
                for item in loc:
                    if isinstance(item, float):
                        item = item*(ratio_width+ratio_height)/2
                    if isinstance(item, list):
                        item = [item[0]*ratio_width, item[1]*ratio_height]
                    item_each.append(item)
                loc_copy.append(item_each)
            self.extra_fields['locs'] = loc_copy

        self.size = size
        if self.masks.ndim<3:  self.masks=self.masks[:,:,None]

        return self

    def transpose(self, method):
        """
            Transpose masks (flip or rotate in 90 degree steps)
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )
        image_width, image_height = self.size
        if method == FLIP_LEFT_RIGHT:
            self.masks = cv2.flip(self.masks,1)
            if 'locs' in self.extra_fields.keys():
                loc_copy = []
                for loc in self.extra_fields['locs']:
                    item_each = []
                    for item in loc:
                        if isinstance(item, list):
                            item[0] = image_width - item[0]- TO_REMOVE
                        item_each.append(item)
                    loc_copy.append(item_each)
                self.extra_fields['locs'] = loc_copy
        elif method == FLIP_TOP_BOTTOM:
            self.masks = cv2.flip(self.masks,0)
            if 'locs' in self.extra_fields.keys():
                loc_copy = []
                for loc in self.extra_fields['locs']:
                    item_each = []
                    for item in loc:
                        if isinstance(item, list):
                            item[1] = image_height - item[1]- TO_REMOVE
                        item_each.append(item)
                    loc_copy.append(item_each)
                self.extra_fields['locs'] = loc_copy
        
        # consider the situation having only one primitive instance
        if self.masks.ndim<3:  self.masks=self.masks[:,:,None]
        
        return self

    def get_binary_seg_target(self, class_index, reshape_size):
        '''
            target: tensor [h, w] (reshape_size)
        '''
        index_list = []
        for i in range(len(self)):
            if self.extra_fields['labels'][i]==class_index:
                index_list.append(i)

        target = self.masks[:,:,index_list].sum(axis=-1)>0
        pad_width = ((0,max(0,reshape_size[0]-self.masks.shape[0])), \
                     (0,max(0, reshape_size[1]-self.masks.shape[1])))
        target=np.pad(target, pad_width, 'constant', constant_values=(False,False)) 

        return target 
        
    def get_inst_seg_target(self, class_index_list, reshape_size):
        '''
            target: list of tensor [w, h] (reshape_size)
        '''
        target = []
        pad_width = ((0,max(0, reshape_size[0]-self.masks.shape[0])), \
                     (0,max(0, reshape_size[1]-self.masks.shape[1])))

        for i in range(len(self)):
            if self.extra_fields['labels'][i] in class_index_list:
                target_each = self.masks[:,:,i]>0
                target.append(
                    np.pad(target_each, pad_width, 'constant', constant_values=(False,False)) 
                )
        return target
                
    def __getitem__(self, item):
        ggeo = GeoList(self.masks[...,item:item+1], self.size)
        for k, v in self.extra_fields.items():
            ggeo.add_field(k, v[item:item+1])
        return ggeo

    def __len__(self):
        return len(self.extra_fields['labels'])

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s

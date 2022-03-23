import numpy as np

class RelList(object):
    
    """
        This class represents a set of relation instances.
        The relation instance are represented as 
        ids: 
            The id of geometric and symbolic primitives (si,pi,li,ci)
        id2nodeid: 
            The dictionary from node ids to node index
        labels_all: 
            Generalized labels of nodes including geometric and symbolic primitives
        labels_text: 
            labels of text nodes (None-text node are -1 and text nodes are corresponding labels in labels_all)
        labels_sem:
            Semantic labels (Different labels of angle, bar, parallel share the same semantic labels)
        edge_mat: 
            Initial sparse matrix of edge, embedding geometric prior knowledge, 
            possible connected edges are 1 and others are 0 
        edge_label: 
            Edge label matrix (symmetric matrix), actual connected edges are 1,
            possible connected but actual not are 0, and impossible connected edges are -1
        conf:
            Probability edge matrix predicted by the model 
    """
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
    CLASSES_SEM = [
        "point","line","circle",
        "text", 
        "perpendicular", "head", "head_len",
        "angle", "bar", "parallel"
    ]

    def __init__(self, ids, labels_all, labels_text):

        self.ids = ids
        self.id2nodeid = dict(zip(ids, range(len(ids)))) 
        self.labels_all = labels_all  
        self.labels_text = labels_text  
        self.labels_sem = self.consturct_semantic_label()  
        self.edge_mat = self.construct_edge_mat() 
        self.edge_label = np.full_like(self.edge_mat, -1) + self.edge_mat 
        
    def consturct_semantic_label(self):
        labels_sem = []
        for id, label in zip(self.ids, self.labels_all):
            if id[0]=='s':
                sym_name = RelList.CLASSES_SYM[label].split(' ')[-1]
                label_sem = RelList.CLASSES_SEM.index(sym_name)
            else:
                geo_name = RelList.CLASSES_GEO[label]
                label_sem = RelList.CLASSES_SEM.index(geo_name)
            labels_sem.append(label_sem)
        return labels_sem

    def construct_edge_mat(self):
        node_num = len(self.ids)
        edge_mat = np.zeros([node_num, node_num])

        for i in range(node_num):
            for j in range(node_num):
                sem_name1 = RelList.CLASSES_SEM[self.labels_sem[i]]
                sem_name2 = RelList.CLASSES_SEM[self.labels_sem[j]]
                for x, y in [(sem_name1, sem_name2), (sem_name2, sem_name1)]:
                    # relation geo with geo 
                    if (x=='point') and (y in ['line', 'circle']) :
                        edge_mat[i,j] = 1
                    # relation geo with sym (excluding head_len which is based on distance rules)
                    if (not x in ['point', 'line', 'circle']) and (y in ['point', 'line', 'circle']):
                        if x in ['text', 'head', 'bar']:
                            edge_mat[i,j] = 1
                        if x=='perpendicular' and y!='circle':
                            edge_mat[i,j] = 1
                        if x=="parallel" and y=='line':
                            edge_mat[i,j] = 1
                        if x=="angle" and y!='cicle':
                            edge_mat[i,j] = 1
                    # relation sym with sym
                    if not (x in ['point', 'line', 'circle'] or y in ['point', 'line', 'circle']):
                        if x=='text' and y in ['head', 'head_len']:
                            edge_mat[i,j] = 1
                        if x=='head' and y=='head_len':
                            edge_mat[i,j] = 1
        return edge_mat       

    def construct_edge_label(self, relation):

        for key in relation:
            for item in relation[key]:
                x = self.id2nodeid[item[0]]
                if key=='geo2geo':
                    y = self.id2nodeid[item[1]]
                    if self.edge_mat[x,y]==1: 
                        self.edge_label[x, y]=1
                        self.edge_label[y, x]=1
                else:
                    for item_s in item[1]:
                        y = self.id2nodeid[item_s]
                        if self.edge_mat[x,y]==1: 
                            self.edge_label[x, y]=1
                            self.edge_label[y, x]=1
    
    def reconstruct_edge_label(self, relation):
        self.edge_label = np.full_like(self.edge_label, 0)
        self.construct_edge_label(relation)

    def construct_edge_label_pred(self, conf, thresh):
        z = np.full_like(self.edge_label, 0)
        z[self.get_edge()] = conf
        # average confidence, directed edges turn into undirected edges
        self.conf = (z+z.T)/2
        self.edge_label += (self.conf>thresh).astype(float)

    def construct_text_label_pred(self, labels, mask):
        t = np.array(self.labels_text)
        t[mask] = labels
        self.labels_text = t.tolist()

    def get_edge(self):
        '''
            get tuple forms of all possible connected edges
        '''
        return np.where(self.edge_mat>0)
    
    def get_edge_label(self):
        '''
            get actual label of possible connected edges
        '''
        return self.edge_label[np.where(self.edge_mat>0)]

    def get_node_label(self):
        return self.labels_text

    def adjust_structure(self, g):
        '''
           Some nodes may have no edges, we should delete them and reconstruct dgl graph before training.
        '''
        node_num = g.num_nodes()
        self.ids_t = self.ids
        self.labels_all_t = self.labels_all
        self.labels_text_t = self.labels_text
        self.labels_sem_t = self.labels_sem
        self.ids = self.ids[:node_num]
        self.labels_all = self.labels_all[:node_num]
        self.labels_text = self.labels_text[:node_num]
        self.labels_sem = self.labels_sem[:node_num] 




import dgl.function as fn
import torch
from torch import nn

class GraphAttention(nn.Module):
    def __init__(self,
             node_dim,
             edge_dim,
             alpha,
             residual=False):
        super(GraphAttention, self).__init__()

        self.fa = nn.Linear(node_dim+edge_dim, edge_dim, bias=False)
        nn.init.xavier_normal_(self.fa.weight.data, gain=1.414)
        self.fnup = nn.Linear(node_dim+edge_dim, node_dim, bias=False)
        nn.init.xavier_normal_(self.fnup.weight.data, gain=1.414)
        self.feup = nn.Linear(node_dim+edge_dim, edge_dim, bias=False)
        nn.init.xavier_normal_(self.feup.weight.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        self.residual = residual

    def forward(self, g, node_feats, edge_feats):
        g.ndata['x'] = node_feats
        g.edata['x'] = edge_feats
        
        g.apply_edges(lambda edges: {'a_p': self.fa(torch.cat((edges.data['x'], edges.dst['x']), 1))})
        g.apply_edges(lambda edges: {'a_p': torch.exp(edges.data['a_p']-torch.max(edges.data['a_p'], dim=0)[0])})
        g.apply_edges(lambda edges: {'a_r': self.fa(torch.cat((edges.data['x'], edges.src['x']), 1))})
        g.apply_edges(lambda edges: {'a_r': torch.exp(edges.data['a_r']-torch.max(edges.data['a_r'], dim=0)[0])})
        g.update_all(fn.copy_edge('a_p', 'a_p'), fn.sum('a_p', 'z'))
        
        g.apply_edges(lambda edges: {'a': edges.data['a_p'] / (edges.dst['z']+1e-5)})
        
        g.edata['a_x'] = g.edata['a'] * g.edata['x']
        g.update_all(fn.copy_edge('a_x', 'a_x'), fn.sum('a_x', 'z'))
        
        ft = torch.cat((g.ndata['z'], g.ndata['x']), 1)
        ft = self.fnup(ft)
        
        g.edata['a_sum'] = g.edata['a_p'] + g.edata['a_r']
        g.edata['a_p'] = g.edata['a_p'] / g.edata['a_sum']
        g.edata['a_r'] = g.edata['a_r'] / g.edata['a_sum']
        g.apply_edges(lambda edges: {'x': torch.cat((edges.data['a_r']*edges.src['x'] + edges.data['a_p']*edges.dst['x'], edges.data['x']), -1)})

        if self.residual:
            g.ndata['x'] = self.leaky_relu(ft) + node_feats
            g.edata['x'] = self.leaky_relu(self.feup(g.edata['x'])) + edge_feats
        else:
            g.ndata['x'] = self.leaky_relu(ft)
            g.edata['x'] = self.leaky_relu(self.feup(g.edata['x']))

        return g.ndata['x'], g.edata['x']

class GAT(nn.Module):
    def __init__(self,
             num_layers=5,
             node_dim=512,
             edge_dim=512,
             node_classes=55,
             edge_classes=2,
             activation=nn.LeakyReLU(0.1),
             alpha=0.1,
             first_residual_layer=1e8):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        assert first_residual_layer > 0
        # input projection (no residual)
        self.gat_layers.append(GraphAttention(
            node_dim, edge_dim, alpha))
        # hidden layers
        for l in range(1, num_layers):
            residual = True if l >= first_residual_layer else False
            self.gat_layers.append(GraphAttention(
                node_dim, edge_dim, alpha, residual))

        # output projection
        self.node_classifier = nn.ModuleList()
        self.node_classifier.append(nn.Linear(node_dim, node_classes, bias=False))
        self.node_classifier = nn.Sequential(*self.node_classifier)
        for m in self.node_classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.414)

        self.edge_classifier = nn.ModuleList()
        self.edge_classifier.append(nn.Linear(edge_dim, edge_classes, bias=False))
        self.edge_classifier = nn.Sequential(*self.edge_classifier)
        for m in self.edge_classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.414)

    def forward(self, g):
        
        node_feats = g.ndata['feats']
        edge_feats = g.edata['feats']
        for l in range(self.num_layers):
            node_feats, edge_feats = self.gat_layers[l](g, node_feats, edge_feats)
            node_feats = self.activation(node_feats)
            edge_feats = self.activation(edge_feats)

        g.ndata['feats'] = node_feats
        g.ndata['confidence'] = self.node_classifier(node_feats)
        g.edata['feats'] = edge_feats
        g.edata['confidence'] = self.edge_classifier(edge_feats)

        return g



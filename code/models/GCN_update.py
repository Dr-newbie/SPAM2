import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatialGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim = 256, out_dim = None, dropout = 0.2):
        super().__init__()
        out_dim = out_dim or in_dim
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops = False, normalize = True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops = False, normalize = True)
        self.dropout = dropout

    def forward(self, x , edge_index, edge_weight = None):
        """
        x: (N, D) node features (이미지 임베딩)
        edge_index: (2, E)
        edge_weight: (E,) or None
        """
        x = x.float()

        if edge_weight is not None:
            edge_weight = edge_weight.float()
            
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x, inplace = True)
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x
            


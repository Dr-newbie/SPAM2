import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class SpatialGCN(nn.Module):
    """
    2-layer GCN over the spatial graph. Node features are the histology
    embeddings H(0); the graph edges come from the spot coordinates.

    normalize:
        True  — each conv recomputes the symmetric D^-1/2 A D^-1/2 coefficients
                from the edge_index it is handed. Correct for a full-graph
                forward pass.
        False — the caller supplies already-normalized edge weights (see
                precompute_gcn_norm). Required for mini-batch training, where a
                subgraph's node degrees are truncated and per-call normalization
                would silently differ from the full-graph result.

    The flag does not change the parameter set, so checkpoints transfer between
    the two modes.
    """

    def __init__(self, in_dim, hidden_dim=256, out_dim=None, dropout=0.2,
                 normalize=True):
        super().__init__()
        out_dim = out_dim or in_dim
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=normalize)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=False, normalize=normalize)
        self.dropout = dropout
        self.normalize = bool(normalize)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: (N, D) node features (histology embeddings)
        edge_index: (2, E)
        edge_weight: (E,) or None. Must be pre-normalized when normalize=False.
        """
        x = x.float()

        if edge_weight is not None:
            edge_weight = edge_weight.float()

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x


@torch.no_grad()
def precompute_gcn_norm(edge_index, edge_weight, num_nodes):
    """
    Symmetric normalization coefficients computed once on the FULL graph.

    Feed the result to a SpatialGCN built with normalize=False. Slicing these
    weights to a subgraph then reproduces the full-graph forward pass exactly,
    which per-call normalization cannot do: gcn_norm derives node degrees from
    whatever edges it is given, so a k-hop subgraph normalizes its boundary
    nodes by a truncated degree.

    Self-loops are assumed to already be present in edge_index (both graph
    builders add them), matching GCNConv(add_self_loops=False).
    """
    ei, ew = gcn_norm(
        edge_index, edge_weight, num_nodes,
        improved=False, add_self_loops=False, flow="source_to_target",
        dtype=torch.float32,
    )
    return ei, ew

import torch.nn as nn
from torch_geometric.nn.models import GAT

class GATLayer(nn.Module):
    def __init__(self, feature_dim, num_layers: int=2):
        super(GATLayer, self).__init__()
        self.gat = GAT(feature_dim, feature_dim, num_layers, feature_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.gat(x, edge_index, edge_weight=edge_weight)
        return x
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class GCN(torch.nn.Module):
    '''
    GCN model implementation using multiple GCNConv layers
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1):
        super().__init__()
        assert num_layers >= 2
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]: 
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index) 
        
        return x

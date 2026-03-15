from torch_geometric.nn import GINConv
from torch.nn import Linear, ReLU, Sequential
import torch
import torch.nn.functional as F


class GIN(torch.nn.Module):
    '''
    GIN model implementation using two GINConv layers
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1, train_eps=True)
        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.conv2 = GINConv(nn2, train_eps=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x        


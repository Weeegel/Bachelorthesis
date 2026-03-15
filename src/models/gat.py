from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F



class GAT(torch.nn.Module):
    '''
    GAT model implementation using two GATConv layers
    
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, heads=5, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index) 
        return x

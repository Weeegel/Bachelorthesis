import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gcn import GCN  
from src.models.gat import GAT
from src.models.gin import GIN
from src.models.graphSage import GraphSAGE


class GraphCLModel(nn.Module):
    '''
    GraphCL model with encoder options and projection head
    in_dim: input feature dimension
    hidden_dim: hidden layer dimension
    proj_dim: projection head output dimension
    num_layers: number of layers in the encoder
    '''
    def __init__(self, in_dim, hidden_dim=256, proj_dim=128, num_layers=2, encoder_type="gcn"):
        super().__init__()

        if encoder_type == "gcn":
            self.encoder = GCN(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        elif encoder_type == "sage":
            self.encoder = GraphSAGE(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=hidden_dim)
        elif encoder_type == "gat":
            self.encoder = GAT(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=hidden_dim)
        elif encoder_type == "gin":
            self.encoder = GIN(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=hidden_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        p = self.projector(z)
        return F.normalize(z, dim=1), F.normalize(p, dim=1)
    
 
    def contrastive_loss(p1, p2, tau=0.4):
        '''
        Computes the contrastive loss between two sets of projections
        p1, p2: projection outputs from two augmented views
        tau: temperature parameter for scaling similarities
          '''
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        sim_matrix = torch.mm(p1, p2.t())  

        pos_sim = sim_matrix.diag()
        numerator = torch.exp(pos_sim / tau)
        denominator = torch.exp(sim_matrix / tau).sum(dim=1)
        loss1 = -torch.log(numerator / denominator).mean()

        sim_matrix_T = sim_matrix.t()
        pos_sim_T = sim_matrix_T.diag()
        numerator_T = torch.exp(pos_sim_T / tau)
        denominator_T = torch.exp(sim_matrix_T / tau).sum(dim=1)
        loss2 = -torch.log(numerator_T / denominator_T).mean()

        return (loss1 + loss2) / 2
import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse
import numpy as np



class LaplacianPosEnc(nn.Module):
    '''
    Computes Laplacian positional encodings and appends them to node features
    k: number of positional encodings to compute
    '''

    def __init__(self, k=16):
        super().__init__()
        self.k = k

    def forward(self, data):
        device = data.x.device if data.x is not None else "cpu"
        num_nodes = data.num_nodes

        lap_edge_index, lap_edge_weight = get_laplacian(data.edge_index, edge_weight=None, normalization='sym', num_nodes=num_nodes)
        
        lap = to_scipy_sparse_matrix(lap_edge_index, lap_edge_weight, num_nodes).tocsc()

        try:
            eigval, eigvec = scipy.sparse.linalg.eigsh(lap, k=self.k + 1, which='SM')
        except Exception:
            lap_dense = torch.tensor(lap.toarray(), dtype=torch.float32)
            eigval, eigvec = torch.linalg.eigh(lap_dense)
            eigval = eigval.cpu().numpy()
            eigvec = eigvec.cpu().numpy()
            idx = np.argsort(eigval)
            eigvec = eigvec[:, idx]

        pos_enc = eigvec[:, 1:self.k + 1]
        pos_enc = torch.tensor(pos_enc, dtype=torch.float32, device=device)

        if data.x is not None:
            data.x = torch.cat([data.x.to(device), pos_enc], dim=1)
        else:
            data.x = pos_enc

        return data

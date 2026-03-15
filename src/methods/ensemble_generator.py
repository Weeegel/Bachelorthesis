import torch
from torch_geometric.data import Data



def generate_ensemble_degree_preserving(data: Data, device, num_swaps: int = None):
    '''
    Generates a degree-preserving random graph by edge swapping
    num_swaps: number of edge swaps to perform
    '''
    
    edge_index = data.edge_index
    row, col = edge_index
    u = torch.min(row, col).cpu()
    v = torch.max(row, col).cpu()
    edges = torch.stack([u, v], dim=1).unique(dim=0).to(device)
    num_edges = edges.size(0)

    if num_swaps is None:
        num_swaps = num_edges

    num_nodes = data.num_nodes

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True

    BATCH = max(num_swaps * 2, 2048)
    all_idx = torch.randint(0, num_edges, (BATCH, 2), device=device)

    done = 0
    batch_pos = 0

    while done < num_swaps:
        if batch_pos >= all_idx.size(0):
            all_idx = torch.randint(0, num_edges, (BATCH, 2), device=device)
            batch_pos = 0

        idx = all_idx[batch_pos]
        batch_pos += 1

        i0, i1 = int(idx[0]), int(idx[1])
        if i0 == i1:
            continue

        u1, v1 = edges[i0, 0], edges[i0, 1]
        u2, v2 = edges[i1, 0], edges[i1, 1]

        a, b = (u1, v2) if u1 < v2 else (v2, u1)
        c, d = (u2, v1) if u2 < v1 else (v1, u2)

        if not (u1 != v1 and u1 != u2 and u1 != v2 and
                v1 != u2 and v1 != v2 and u2 != v2):
            continue
        if a == b or c == d:
            continue
        if adj[a, b] or adj[c, d]:
            continue

        edges[i0] = torch.tensor([a, b], device=device)
        edges[i1] = torch.tensor([c, d], device=device)
        adj[u1, v1] = adj[v1, u1] = False
        adj[u2, v2] = adj[v2, u2] = False
        adj[a, b]   = adj[b, a]   = True
        adj[c, d]   = adj[d, c]   = True

        done += 1

    new_edges = edges.t()
    new_edge_index = torch.cat([new_edges, new_edges[[1, 0]]], dim=1)
    return Data(x=data.x, edge_index=new_edge_index, y=data.y)



def generate_ensemble_full_random(data: Data, device, num_swaps: int = None):
    ''' 
    Generates a fully random graph by randomly sampling edges
    num_swaps: number of edges to swap (i.e., number of new random edges to add)
    '''
    num_nodes = data.num_nodes
    total_edges = data.edge_index.size(1) // 2
    
    if num_swaps is None:
        num_swaps = total_edges
    
    perm = torch.randperm(total_edges)
    keep = perm[num_swaps:]
    kept_edges = data.edge_index[:, keep]
    
    new_edges = set()
    while len(new_edges) < num_swaps:
        u = torch.randint(0, num_nodes, ()).item()
        v = torch.randint(0, num_nodes, ()).item()
        if u != v:
            new_edges.add((u, v))
    
    row, col = zip(*new_edges)
    new_edge_index = torch.tensor([row + col, col + row], device=device)
    edge_index = torch.cat([kept_edges, new_edge_index], dim=1)
    return Data(x=data.x, edge_index=edge_index, y=data.y)
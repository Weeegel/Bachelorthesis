import random 
import torch 


def aug_random_mask(x, drop_percent=0.2):
    '''
    Randomly masks a percentage of node features by setting them to zero
    x: node feature matrix [num_nodes, num_features]
    drop_percent: percentage of features to mask
    '''
    _, num_feats = x.size()
    mask_num = int(num_feats * drop_percent)
    if mask_num == 0:
        return x.clone()

    mask_idx = random.sample(range(num_feats), mask_num)
    aug_x = x.clone()
    aug_x[:, mask_idx] = 0.0
    return aug_x


def aug_random_edge(edge_index, num_nodes, device, drop_percent=0.2):
    """ 
    Randomly drops a percentage of edges and adds the same number of random edges
    edge_index: [2, num_edges]
    """ 
    E = edge_index.size(1) 
    drop_num = int(E * drop_percent) 
    if drop_num == 0: 
        return edge_index.clone() 
    perm = torch.randperm(E, device=device) 
    keep_mask = perm[drop_num:] 
    edge_index_dropped = edge_index[:, keep_mask] 
    num_add = drop_num 
    src_rand = torch.randint(0, num_nodes, (num_add,), device=device) 
    dst_rand = torch.randint(0, num_nodes, (num_add,), device=device) 
    mask_selfloop = src_rand != dst_rand 
    src_rand, dst_rand = src_rand[mask_selfloop], dst_rand[mask_selfloop] 
    added = torch.stack([src_rand, dst_rand], dim=0) 
    edge_index_aug = torch.cat([edge_index_dropped, added], dim=1) 
    return edge_index_aug 

def aug_feature_noise(x, device, noise_std=0.1): 
    """ 
    Add Gaussian noise to node features.
    x: [num_nodes, num_features]
    noise_std: standard deviation of the Gaussian noise
    """ 
    noise = torch.randn_like(x, device=device) * noise_std 
    return x + noise 

def get_augmented_view(data,device, drop_feat=0.1, drop_edge=0.1, noise_std=0.1): 
    '''
    Generates an augmented view of the graph by applying random feature masking, edge perturbation, and feature noise
    '''
    x = data.x 
    edge_index = data.edge_index 
    x = aug_random_mask(x, drop_percent=drop_feat) 
    if noise_std > 0:
        x = aug_feature_noise(x, device=device, noise_std=noise_std)

    edge_index = aug_random_edge(edge_index, num_nodes=x.size(0),device=device,drop_percent= drop_edge) 
    return x, edge_index.long()
import torch
from src.models.graphCL_model import GraphCLModel
import torch
import torch.nn.functional as F
from src.plotting import visualize_embeddings_grid
from src.methods.randomPerturbations import get_augmented_view
from src.methods.ensemble_generator import generate_ensemble_degree_preserving, generate_ensemble_full_random


def train_graphcl(data, encoders, device, mode="rand_pert", contrastive_epochs=50, batch_size=4096, tau=0.4, lr=1e-3, num_swaps=1500, drop_feat=0.1, drop_edge=0.1, noise_std=0.1, visualize=True, use_batching=True):
    '''
    Trains GraphCL models with various encoders and augmentation modes
    Returns trained models, their embeddings, and loss history
    Uses different augmentation strategies based on the specified mode
    '''
    loss_history = {}
    data = data.to(device)
    num_nodes = data.num_nodes
    
    models = {enc: GraphCLModel(in_dim=data.x.size(1), encoder_type=enc).to(device) for enc in encoders}
    optims = {enc: torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) for enc, model in models.items()}
    all_embeddings = {}
    
    for encoder_name, model in models.items():
        loss_history[encoder_name] = []
        optimizer = optims[encoder_name]
        print(f"\nTraining GraphCL Encoder: {encoder_name.upper()} ({mode})")
    
        for epoch in range(contrastive_epochs):
            model.train()
            
            x1, e1 = generate_view(data, device, mode=mode, num_swaps=num_swaps,drop_feat=drop_feat, drop_edge=drop_edge, noise_std=noise_std)
            x1 = F.normalize(x1, p=2, dim=-1)
            
            x2, e2 = generate_view(data, device, mode=mode, num_swaps=num_swaps,drop_feat=drop_feat, drop_edge=drop_edge, noise_std=noise_std)
            x2 = F.normalize(x2, p=2, dim=-1)
            
            _, p1 = model(x1, e1)
            _, p2 = model(x2, e2)
            
            p1 = F.normalize(p1, dim=1)
            p2 = F.normalize(p2, dim=1)
            
            if use_batching and num_nodes > batch_size:
                perm = torch.randperm(num_nodes, device=device)
                total_loss = 0.0
                num_batches = 0
                
                for i in range(0, num_nodes, batch_size):
                    idx = perm[i:i + batch_size]
                    batch_loss = GraphCLModel.contrastive_loss(p1[idx], p2[idx], tau=tau)
                    total_loss += batch_loss
                    num_batches += 1
                
                loss = total_loss / num_batches
            else:
                loss = GraphCLModel.contrastive_loss(p1, p2, tau=tau)

            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_history[encoder_name].append(loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | GraphCL Loss: {loss.item():.4f}")
        
        model.eval()
        with torch.inference_mode():
            x_normalized = F.normalize(data.x, p=2, dim=-1)
            final_embeddings, _ = model(x_normalized, data.edge_index)
            all_embeddings[encoder_name] = final_embeddings.cpu()
        
        
    if visualize:
            visualize_embeddings_grid(all_embeddings, data.y.cpu())
    return models, all_embeddings, loss_history


def generate_view(data, device, mode="Rand-Aug",num_swaps=1500, drop_feat=0.1, drop_edge=0.1,noise_std =0.1):
    if mode == "Rand-Aug":
        x, edge_index = get_augmented_view(data,device=device,drop_feat=drop_feat,drop_edge=drop_edge, noise_std=noise_std)
        return x, edge_index

    elif mode == "DP-Rewire":
        g = generate_ensemble_degree_preserving(data, device=device, num_swaps=num_swaps)
        return g.x, g.edge_index

    elif mode == "DP-Aug":
        g = generate_ensemble_degree_preserving(data,device=device, num_swaps=num_swaps)
        x, edge_index = get_augmented_view( g,device=device, drop_feat=drop_feat, drop_edge=drop_edge, noise_std=noise_std)
        return x, edge_index

    elif mode == "Rand-Rewire":
        g = generate_ensemble_full_random(data, device=device,num_swaps=num_swaps)
        return g.x, g.edge_index
    
    elif mode == "none":
        return data.x, data.edge_index
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

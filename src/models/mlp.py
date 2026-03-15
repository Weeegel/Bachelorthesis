import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.plotting import plot_confusion_matrices
import matplotlib.cm as cm



class MLP(nn.Module):
    '''
    Multi-layer Perceptron for node classification on embeddings
    '''
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


    def evaluate_with_mlp(embeddings,data, device, hidden_dims1=[128], hidden_dims2=[128], dropouts=[0.0,0.1, 0.2], lrs=[0.0005, 0.001,0.002],weight_decays=[1e-5,1e-4, 5e-4], mlp_epochs=100):
        '''
        Evaluates the quality of embeddings with an MLP classifier.
        '''
        labels = data.y.to(device)

        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        val_idx = data.val_mask.nonzero(as_tuple=True)[0]
        test_idx = data.test_mask.nonzero(as_tuple=True)[0]

        num_classes = int(labels.max().item() + 1)
        best_results = {}
        best_curves = {}
        best_models = {}
        best_test_data = {}

        for emb_name, z in embeddings.items():
            print(f"\nMLP on {emb_name.upper()} embeddings")

            z = F.normalize(z.to(device), dim=1)

            x_train, y_train = z[train_idx], labels[train_idx]
            x_val, y_val = z[val_idx], labels[val_idx]
            x_test, y_test = z[test_idx], labels[test_idx]

            best_val_overall = 0.0
            best_cfg = None

            for h1 in hidden_dims1:
                for h2 in hidden_dims2:
                    for d in dropouts:
                        for lr in lrs:
                            for wd in weight_decays:
                                model = MLP(in_dim=z.size(1), hidden_dim1=h1, hidden_dim2=h2, out_dim=num_classes, dropout=d).to(device)
                                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                                criterion = nn.CrossEntropyLoss()

                                best_val_acc = 0.0
                                best_state = None

                                train_accs = []
                                val_accs = []

                                for _ in range(mlp_epochs):
                                    _, train_acc = train_epoch(model, x_train, y_train, optimizer, criterion)
                                    val_acc = eval_acc(model, x_val, y_val)
                                    train_accs.append(train_acc)
                                    val_accs.append(val_acc)

                                    if val_acc > best_val_acc:
                                        best_val_acc = val_acc
                                        best_state = {k: v.clone() for k, v in model.state_dict().items()}

                                if best_val_acc > best_val_overall:
                                    best_val_overall = best_val_acc
                                    best_cfg = (h1, h2, d, lr, wd)

                                    model.load_state_dict(best_state)
                                    best_models[emb_name] = model
                                    best_test_data[emb_name] = (x_test, y_test)
                                    best_curves[emb_name] = {"train": train_accs,"val": val_accs}

                                print(f"[{emb_name.upper()}] h1={h1}, h2={h2}, d={d}, lr={lr}, wd={wd} | Val {best_val_acc:.4f}")

            model = best_models[emb_name]
            x_test, y_test = best_test_data[emb_name]
            final_test_acc = eval_acc(model, x_test, y_test)

            best_results[emb_name] = (final_test_acc, best_cfg)
            print(f"Best Test Accuracy ({emb_name.upper()}): {final_test_acc:.4f}")


        best_acc_per_encoder = {emb_name: best_test_acc for emb_name, (best_test_acc, _) in best_results.items()}
        print("\nBest configurations per encoder:")
        for emb_name, (acc, cfg) in best_results.items():
            h1, h2, d, lr, wd = cfg
            print(f"{emb_name.upper():4s} | "f"acc={acc:.4f} | " f"h1={h1}, h2={h2}, d={d}, lr={lr}, wd={wd}")
        
        encoder_names = list(best_curves.keys())
        num_encoders = len(encoder_names)
        colors = cm.get_cmap("tab10", num_encoders)  

        plt.figure(figsize=(8,5))

        for i, emb_name in enumerate(encoder_names):
            curves = best_curves[emb_name]
            color = colors(i)
            plt.plot(curves["val"], label=f"{emb_name} (val)", color=color)
            plt.plot(curves["train"], linestyle="--", alpha=0.6, label=f"{emb_name} (train)", color=color)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("MLP Training Curves (best configuration)")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

        plot_confusion_matrices(best_models, best_test_data, suptitle="MLP – Confusion Matrices")

        return best_results, best_acc_per_encoder


def train_epoch(model, x, y, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    acc = (out.argmax(dim=1) == y).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def eval_acc(model, x, y):
    model.eval()
    pred = model(x).argmax(dim=1)
    return (pred == y).float().mean().item()
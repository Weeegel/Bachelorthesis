import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.functional import cosine_similarity
import os
from pathlib import Path
import torch
import pandas as pd
from IPython.display import display, Markdown


def visualize_embeddings(emb, labels, title, seed=42):
    emb = emb.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    tsne = TSNE(n_components=2,perplexity=30,random_state=seed,init="pca",learning_rate="auto")
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(emb_2d[:, 0],emb_2d[:, 1],c=labels,cmap="tab10",s=10,alpha=0.8)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def visualize_embeddings_grid(all_embeddings, labels, title="Embeddings", seed=42):
    DISTINCT_COLORS = [
        "#4363d8",  
        "#e6194b",  
        "#3cb44b",  
        "#ffe119",
        "#911eb4",
        "#f032e6",
        "#f58231",
        "#42d4f4",
        "#9a6324",
        "#000075", 
    ]

    labels_np = labels.cpu().numpy()
    unique_classes = np.unique(labels_np)
    color_map = {cls: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, cls in enumerate(unique_classes)}
    point_colors = [color_map[l] for l in labels_np]

    encoder_names = list(all_embeddings.keys())
    n = len(encoder_names)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))
    axes = axes.flatten()

    tsne = TSNE(n_components=2, perplexity=30, random_state=seed,init="pca", learning_rate="auto")

    for i, enc_name in enumerate(encoder_names):
        emb = all_embeddings[enc_name].detach().cpu().numpy()
        emb_2d = tsne.fit_transform(emb)
        axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=point_colors, s=10, alpha=0.8)
        axes[i].set_title(enc_name.upper())
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_loss_per_encoder(all_losses, encoders=["gcn", "sage", "gat", "gin"]):
    for enc in encoders:
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, enc_losses in all_losses.items():
            if enc in enc_losses:
                loss = enc_losses[enc]
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss)
                ax.plot(loss.numpy(), label=mode)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Contrastive Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"loss_{enc}.pdf", bbox_inches="tight", dpi=300)
        plt.show()

    
@torch.no_grad()
def plot_confusion_matrix(model, x_test, y_test, title):
    if hasattr(model, "eval"):
        model.eval()
        with torch.no_grad():
            preds = model(x_test).argmax(dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    else:
        preds = model.predict(x_test.cpu().numpy())
        y_true = y_test.cpu().numpy()

    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_confusion_matrices(models_dict, test_data_dict, suptitle="Confusion Matrices"):
    n = len(models_dict)
    ncols = 2
    nrows = (n + 1) // 2  
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, (emb_name, model) in enumerate(models_dict.items()):
        x_test, y_test = test_data_dict[emb_name]

        if hasattr(model, "eval"):
            model.eval()
            with torch.no_grad():
                preds = model(x_test).argmax(dim=1).cpu().numpy()
        else:
            preds = model.predict(x_test.cpu().numpy())

        y_true = y_test.cpu().numpy()

        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(emb_name.upper())

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_accs_bar_with_singe_embs(other_accs, modes, encoders, accs):
    x_modes = np.arange(len(modes))
    width = 0.18

    x_single = np.arange(len(other_accs)) + len(modes) + 1

    plt.figure(figsize=(14, 6))

    for i, enc in enumerate(encoders):
        heights = [accs[mode]["mean"].get(enc, np.nan)for mode in modes]
        stds  = [accs[mode]["std"].get(enc, 0) for mode in modes]
        plt.bar(x_modes + i * width, heights, width,yerr=stds, label=enc.upper())

    plt.bar(x_single, list(other_accs.values()), width, color="gray", alpha=0.8, label="Single embeddings")

    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    xticks = list(x_modes + (len(encoders) - 1) / 2 * width) + list(x_single)
    xlabels = modes + list(other_accs.keys())

    plt.xticks(xticks, xlabels, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_bar.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_loss_vs_acc_per_encoder(encoders, all_losses, accs):
    for enc in encoders:
        final_losses = []
        final_accs = []
        labels = []

        for mode in all_losses.keys():
            loss_curve = torch.tensor(all_losses[mode][enc])
            final_losses.append(loss_curve[-1].item())
            final_accs.append(accs[mode]["mean"][enc])
            labels.append(mode)

        plt.figure(figsize=(6, 5))
        plt.scatter(final_losses, final_accs)

        for x, y, label in zip(final_losses, final_accs, labels):
            plt.text(x, y, label, fontsize=9)

        plt.xlabel("Final Contrastive Loss")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"loss_vs_acc_{enc}.pdf", bbox_inches="tight", dpi=300)
        plt.show()


def cosine_similarity_matrix(emb, num_samples=10000, seed=42):
    torch.manual_seed(seed)
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    num_nodes = emb.size(0)

    idx_i = torch.randint(0, num_nodes, (num_samples,))
    idx_j = torch.randint(0, num_nodes, (num_samples,))
    
    sims = (emb[idx_i] * emb[idx_j]).sum(dim=1)

    return idx_i, idx_j, sims

def intra_inter_similarities_sampled(emb, labels, num_samples=10000):
    idx_i, idx_j, sims = cosine_similarity_matrix(emb, num_samples)
    labels_i = labels[idx_i]
    labels_j = labels[idx_j]

    intra = sims[labels_i == labels_j]
    inter = sims[labels_i != labels_j]
    return intra, inter

def plot_intra_inter(intra, inter, title):
    plt.figure(figsize=(6, 5))
    plt.boxplot([intra.numpy(), inter.numpy()], tick_labels=['Intra-class', 'Inter-class'])
    plt.ylabel('Cosine similarity')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_avgAcc_vs_singe_embs(modes, other_accs, accs,title):
    ensemble_accs = {
        mode: np.nanmean(list(accs[mode]["mean"].values()))
        for mode in modes
    }

    labels = list(ensemble_accs.keys()) + list(other_accs.keys())
    values = list(ensemble_accs.values()) + list(other_accs.values())

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x,values,color=["tab:blue"] * len(ensemble_accs) + ["gray"] * len(other_accs),alpha=0.8)

    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def save_results(mode: str,encoders: list,all_run_losses: list,all_run_embeddings: list,all_run_accs: list,all_run_svm: list,filenames: dict):
    avg_losses = {}
    avg_embeddings = {}
    avg_acc = {}
    std_acc = {}
    avg_svm = {}
    std_svm = {}

    for enc in encoders:
        curves = [torch.tensor(run[enc]) for run in all_run_losses]
        avg_losses[enc] = torch.stack(curves).mean(dim=0)
    
    for enc in encoders:
        embs = [run[enc] for run in all_run_embeddings]
        avg_embeddings[enc] = torch.stack(embs).mean(dim=0)
    
    for enc in encoders:
        vals = [run[enc] for run in all_run_accs]
        avg_acc[enc] = float(np.mean(vals))
        std_acc[enc] = float(np.std(vals))
    
    for enc in encoders:
        values = [run[enc] for run in all_run_svm]
        avg_svm[enc] = float(np.mean(values))
        std_svm[enc] = float(np.std(values))

    if os.path.exists(filenames['loss']):
        all_losses = torch.load(filenames['loss'])
    else:
        all_losses = {}
    all_losses[mode] = avg_losses
    torch.save(all_losses, filenames['loss'])
    
    if os.path.exists(filenames['emb']):
        all_embeddings = torch.load(filenames['emb'])
    else:
        all_embeddings = {}
    all_embeddings[mode] = {enc: emb.detach().cpu() for enc, emb in avg_embeddings.items()}
    torch.save(all_embeddings, filenames['emb'])
    
    if os.path.exists(filenames['acc']):
        all_accs = torch.load(filenames['acc'])
    else:
        all_accs = {}
    all_accs[mode] = {"mean": avg_acc, "std": std_acc, "runs": all_run_accs}
    torch.save(all_accs, filenames['acc'])
    
    if os.path.exists(filenames['svm']):
        all_svm = torch.load(filenames['svm'])
    else:
        all_svm = {}
    all_svm[mode] = {"mean": avg_svm, "std": std_svm, "runs": all_run_svm}
    torch.save(all_svm, filenames['svm'])
    
    return avg_losses, avg_embeddings, avg_acc, std_acc, avg_svm, std_svm


def display_result_tables(accs, svm_results, other_accs, svm_results_node_pos):
    def highlight_max_col(df, parse_pm=False):
        numeric = df.copy()

        if parse_pm:
            for col in df.columns[1:]:
                numeric[col] = df[col].apply(lambda v: float(v.split("±")[0]) if v != "n/a" else -1)
        else:
            numeric.iloc[:, 1:] = numeric.iloc[:, 1:].astype(float)

        def highlight(col):
            max_val = col.max()
            return ['font-weight: bold' if v == max_val else '' for v in col]

        return df.style.apply(highlight, axis=0, subset=df.columns[1:])

    modes = list(accs.keys())
    encoders = list(next(iter(accs.values()))["mean"].keys())

    # MLP
    mlp_rows = []
    for mode in modes:
        row = {"Mode": mode}
        for enc in encoders:
            mean = accs[mode]["mean"].get(enc, float('nan'))
            std  = accs[mode]["std"].get(enc, float('nan'))
            row[enc] = f"{mean:.3f} ± {std:.3f}" if not pd.isna(mean) else "n/a"
        mlp_rows.append(row)

    df_mlp = pd.DataFrame(mlp_rows)
    display(Markdown("MLP Results"))
    display(highlight_max_col(df_mlp, parse_pm=True))

    # SVM
    svm_rows = []
    for mode in modes:
        row = {"Mode": mode}
        for enc in encoders:
            if mode in svm_results:
                mean = svm_results[mode]["mean"].get(enc, float('nan'))
                std  = svm_results[mode]["std"].get(enc, float('nan'))
                row[enc] = f"{mean:.3f} ± {std:.3f}" if not pd.isna(mean) else "n/a"
            else:
                row[enc] = "n/a"
        svm_rows.append(row)

    df_svm = pd.DataFrame(svm_rows)
    display(Markdown("SVM Results"))
    display(highlight_max_col(df_svm, parse_pm=True))

    # baselines
    df_special_rows = []
    for emb_name in other_accs.keys():
        row = {"Embedding": emb_name}

        mlp_acc = other_accs.get(emb_name, float('nan'))
        row["MLP Test Accuracy"] = f"{mlp_acc:.3f}" if not pd.isna(mlp_acc) else "n/a"

        svm_acc = svm_results_node_pos.get(emb_name, float('nan'))
        row["SVM Test Accuracy"] = f"{svm_acc:.3f}" if not pd.isna(svm_acc) else "n/a"

        df_special_rows.append(row)

    df_special = pd.DataFrame(df_special_rows)
    display(Markdown("Node2Vec & PosEnc & Original Features – MLP + SVM"))
    display(highlight_max_col(df_special))

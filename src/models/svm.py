from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from src.plotting import plot_confusion_matrix

def evaluate_with_svm(embeddings, data, device, Cs=[0.01, 0.1, 1, 10]):
    '''
    Evaluates the quality of embeddings with SVM classifier.
    '''
    labels = data.y.cpu().numpy()

    train_idx = data.train_mask[:, 0].cpu().numpy()
    val_idx = data.val_mask[:, 0].cpu().numpy()
    test_idx = data.test_mask[:, 0].cpu().numpy()

    best_results = {}
    all_test_accs = {}

    for emb_name, z in embeddings.items():
        print(f"\nSVM on {emb_name.upper()} embeddings")

        z = F.normalize(z, dim=1).cpu().numpy()

        X_train, y_train = z[train_idx], labels[train_idx]
        X_val, y_val = z[val_idx], labels[val_idx]
        X_test, y_test = z[test_idx], labels[test_idx]

        best_val_acc = 0
        best_model = None
        best_C = None

        for C in Cs:
            clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000)
            clf.fit(X_train, y_train)
            val_pred = clf.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)

            print(f"C={C} | Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = clf
                best_C = C

        test_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"Best C={best_C} | Test Acc={test_acc:.4f}")

        best_results[emb_name] = test_acc
        all_test_accs[emb_name] = test_acc

        plot_confusion_matrix(best_model, torch.tensor(X_test), torch.tensor(y_test), title=f"Confusion Matrix – {emb_name}")

    plt.figure(figsize=(6,4))
    plt.bar(all_test_accs.keys(), all_test_accs.values())
    plt.ylabel("Test Accuracy")
    plt.title("SVM Performance on Embeddings")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

    return best_results

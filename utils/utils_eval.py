import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    average_precision_score
)


def remap_labels(labels):
    """
    Remap labels to a contiguous range starting from 0.
    Handles both 1D and 2D tensors.
    """
    labels = labels.flatten()  # Flatten the tensor to ensure 1D
    unique_labels = torch.unique(labels)
    label_map = {old_label.item(): new_idx for new_idx, old_label in enumerate(unique_labels)}
    remapped_labels = torch.tensor([label_map[label.item()] for label in labels], dtype=torch.long)
    return remapped_labels.view_as(labels)  # Reshape to original shape



# Custom PCA scatter plot using the provided function
def pca_scatter_plot(support_embeddings, query_embeddings, support_labels, query_labels, filename):
    """
    Save a PCA scatter plot showing both support and query embeddings.
    
    Parameters:
    - support_embeddings: Embeddings of the support set
    - query_embeddings: Embeddings of the query set
    - support_labels: Labels for the support set
    - query_labels: Labels for the query set
    - filename: File path to save the plot
    """
    # Perform PCA on concatenated embeddings
    all_embeddings = np.vstack((support_embeddings, query_embeddings))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeddings)

    # Split PCA results into support and query
    support_pca = pca_result[:len(support_embeddings)]
    query_pca = pca_result[len(support_embeddings):]

    # Unique classes
    classes = np.unique(np.concatenate((support_labels, query_labels)))

    # Assign colors to each class
    colors = {cls: f"C{idx}" for idx, cls in enumerate(classes)}

    # Plot Support Set Embeddings
    plt.figure(figsize=(10, 8))
    for cls in classes:
        cls_indices = np.where(support_labels == cls)[0]
        plt.scatter(
            support_pca[cls_indices, 0],
            support_pca[cls_indices, 1],
            color=colors[cls],
            label=f"Support Class {cls}",
            edgecolor="k",
            alpha=0.8,
            s=80,
        )

    # Plot Query Set Embeddings
    for cls in classes:
        cls_indices = np.where(query_labels == cls)[0]
        plt.scatter(
            query_pca[cls_indices, 0],
            query_pca[cls_indices, 1],
            color=colors[cls],
            label=f"Query Class {cls}",
            edgecolor="k",
            alpha=0.6,
            s=80,
            marker="^",  # Different marker for query
        )

    # Add legend, title, and labels
    plt.legend()
    plt.title("Support and Query Embeddings in 2D Space (Class-Specific Colors)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)

    # Save the plot
    plt.savefig(filename)
    plt.close()



def compute_metrics(y_true, y_pred, y_prob, n_way):
    """
    Compute various classification metrics for multiclass tasks using macro averaging.

    Parameters
    ----------
    y_true : 1D array-like
        True labels.
    y_pred : 1D array-like
        Predicted labels.
    y_prob : 2D array-like (N x C)
        Probability predictions for each class.
    n_way : int
        Number of classes.

    Returns
    -------
    metrics_dict : dict
        Dictionary containing overall metrics:
        - accuracy
        - precision (macro)
        - recall (macro)
        - f1 (macro)
        - roc_auc (macro-ovr if n_way > 2; otherwise binary)
        - average_precision (macro)
        - mcc (multiclass)
    per_class : dict
        Dictionary containing per-class metrics:
        - accuracy, precision, recall, f1, average_precision
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Macro P/R/F1
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)  # multiclass MCC

    # AUC (multiclass macro-OvR for n_way > 2, or standard for binary)
    if n_way > 2:
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = None
    else:
        # For binary, do standard roc_auc
        roc_auc = roc_auc_score(y_true, y_prob[:, 1]) if n_way == 2 else None

    # Macro average precision (AUC-PR)
    # Binarize the labels for multiclass
    y_true_binarized = label_binarize(y_true, classes=np.arange(n_way))
    try:
        avg_precision = average_precision_score(
            y_true_binarized, y_prob, average="macro"
        )
    except ValueError:
        avg_precision = None

    # Compile overall metrics
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "mcc": mcc
    }

    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred, labels=range(n_way))
    per_class_acc = cm.diagonal() / np.where(cm.sum(axis=1) == 0, 1, cm.sum(axis=1))

    per_class_precision = precision_score(y_true, y_pred, labels=range(n_way), average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=range(n_way), average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=range(n_way), average=None, zero_division=0)

    # Per-class average precision (one-vs-rest)
    per_class_ap = []
    if y_true_binarized.shape[1] == n_way:
        for class_idx in range(n_way):
            try:
                ap_score = average_precision_score(y_true_binarized[:, class_idx], y_prob[:, class_idx])
            except ValueError:
                ap_score = None
            per_class_ap.append(ap_score)
    else:
        per_class_ap = [None]*n_way

    per_class = {
        "accuracy": per_class_acc,
        "precision": per_class_precision,
        "recall": per_class_recall,
        "f1": per_class_f1,
        "average_precision": per_class_ap
    }

    return metrics_dict, per_class


def plot_confusion_matrix(y_true, y_pred, display_labels, save_path, title="Confusion Matrix"):
    """
    Saves confusion matrix to disk.
    """
    cm = confusion_matrix(y_true, y_pred, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues')
    disp.ax_.set_title(title)
    disp.figure_.savefig(save_path)
    disp.figure_.clf()


def evaluate_on_test(
    classifier, model_biot,
    test_query_x, test_query_y,
    device, n_way, criterion
):
    """
    Perform final evaluation on the test query set and compute metrics.
    Should return:
    - metrics_dict (macro-F1, precision, etc.)
    - per_class (per-class metrics)
    - test_loss
    - y_true (true labels)
    - y_pred (naive argmax predictions)
    - test_probs (softmax probabilities for thresholding)
    """
    classifier.eval()
    with torch.no_grad():
        test_query_y_mapped = remap_labels(test_query_y).to(device)
        test_query_x_gpu = test_query_x.to(device)

        # Forward pass through model
        test_query_emb = model_biot(test_query_x_gpu.squeeze(0))
        test_logits = classifier(test_query_emb)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()  # softmax probs

        _, test_preds = torch.max(test_logits, dim=1)  # argmax predictions
        test_loss = criterion(test_logits, test_query_y_mapped.squeeze(0))

        y_true = test_query_y_mapped.squeeze(0).cpu().numpy()
        y_pred = test_preds.cpu().numpy()

        # Compute standard metrics (argmax-based)
        metrics_dict, per_class = compute_metrics(
            y_true, y_pred, test_probs, n_way=n_way
        )

    # Must return all 6 values!
    return metrics_dict, per_class, test_loss.item(), y_true, y_pred, test_probs



def generate_report_and_plots(results, save_dir):
    """
    Generate a formatted report and plots based on the results.

    Parameters
    ----------
    results : dict
        Dictionary containing metrics for all models and k_shot variations.
    save_dir : str
        Directory to save the report and plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    for model_name, metrics_per_kshot in results.items():
        report = []
        all_per_class_accuracies = []

        for k_shot, (aggregated_metrics_mean, aggregated_metrics_std) in metrics_per_kshot.items():
            # (No change for overall metrics.)
            row_dict = {"k_shot": k_shot}
            
            for key in aggregated_metrics_mean:
                mean_val = aggregated_metrics_mean[key]
                std_val = aggregated_metrics_std[key]
                if mean_val is not None and std_val is not None:
                    row_dict[key] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    row_dict[key] = "N/A"
            report.append(row_dict)


        # Convert to DataFrame for overall metrics
        report_df = pd.DataFrame(report)
        # Save
        report_df.to_csv(os.path.join(save_dir, f"{model_name}_results.csv"), index=False)


        if "accuracy" in report_df.columns:
            plt.figure(figsize=(10, 6))
            x_vals = report_df["k_shot"].values
            y_means = []
            y_stds = []
            
            for val in report_df["accuracy"]:
                if val == "N/A":
                    y_means.append(np.nan)
                    y_stds.append(0)
                else:
                    mean_str, std_str = val.split(" ± ")
                    y_means.append(float(mean_str))
                    y_stds.append(float(std_str))
        
            plt.errorbar(
                x_vals, y_means, yerr=y_stds, fmt='-o',
                capsize=5, elinewidth=2, markeredgewidth=2,
                label=f"{model_name.upper()} - Accuracy"
            )
            
            plt.title(f"{model_name.upper()} Performance Across k_shot ")
            plt.xlabel("Number of Shots (k_shot)")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
        
            # Set x-axis to show only full integers
            x_ticks = np.unique(x_vals.astype(int))  # Ensure unique integer values
            plt.xticks(x_ticks)
        
            plt.savefig(os.path.join(save_dir, f"{model_name}_performance.png"))
            plt.close()


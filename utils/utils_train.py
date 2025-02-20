import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import FCNet, TransformerClassifier
import os
import h5py
from tqdm import tqdm
from scipy.signal import resample
import faiss
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


from utils.utils_eval import plot_confusion_matrix, evaluate_on_test, compute_metrics, generate_report_and_plots


class FewShotDataset(Dataset):
    """
    Few-Shot Learning Dataset for EEG signals.
    
    This dataset supports episodic sampling for few-shot learning, ensuring class-balanced support and query sets.
    
    Parameters
    ----------
    root : str
        Root directory containing EEG data files.
    files : list
        List of filenames available for sampling.
    n_way : int, optional
        Number of classes per episode (default is 3).
    k_shot : int, optional
        Number of support examples per class (default is 3).
    q_query : int, optional
        Number of query examples per class (default is 20).
    test_q_query : int, optional
        Number of test query examples per class (default is 20).
    sampling_rate : int, optional
        Desired sampling rate for EEG signals (default is 200 Hz).
    default_rate : int, optional
        Default rate of EEG signals in the dataset (default is 256 Hz).
    num_episodes : int, optional
        Number of episodes for training/evaluation (default is 5).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    """
    def __init__(self, root, files, n_way=3, k_shot=3, q_query=20, test_q_query=20,
                 sampling_rate=200, default_rate=256, num_episodes=5, seed=42):
        self.root = root
        self.files = files
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.test_q_query = test_q_query
        self.sampling_rate = sampling_rate
        self.default_rate = default_rate
        self.num_episodes = num_episodes
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Group files by class and patient
        self.class_to_patient_files = {}
        for f in self.files:
            label = self._get_label(os.path.join(self.root, f))
            if label in [2, 3, 4]:  # Adjust classes as needed
                pid = self._get_patient_id(f)
                self.class_to_patient_files.setdefault(label, {}).setdefault(pid, []).append(f)

        self.classes = sorted(list(self.class_to_patient_files.keys()))

        # Predefine the set of patients for support
        self.predefined_support_patients = self._prepare_predefined_support_patients()

        # Precompute fixed query and test query sets
        self.fixed_query, self.fixed_test_query = self._prepare_fixed_query_and_test_query()

    def __len__(self):
        return self.num_episodes

    def _prepare_predefined_support_patients(self):
        """Predefine support patients for each class."""
        predefined_support = {}
        for c in self.classes:
            pids = sorted(self.class_to_patient_files[c].keys())
            predefined_support[c] = random.sample(pids, min(len(pids), self.num_episodes))
        return predefined_support

    def _prepare_fixed_query_and_test_query(self):
        """Prepare fixed query and test query sets."""
        query_x, query_y = [], []
        test_query_x, test_query_y = [], []

        used_patients = set()
        for c in self.classes:
            pids = sorted(self.class_to_patient_files[c].keys())
            query_pids = random.sample([pid for pid in pids if pid not in used_patients], 5)
            used_patients.update(query_pids)

            q_files = []
            for qpid in query_pids:
                q_files.extend(self.class_to_patient_files[c][qpid])
            chosen_q_files = q_files[:self.q_query]

            for qf in chosen_q_files:
                Xq = self._load_and_preprocess(os.path.join(self.root, qf))
                query_x.append(Xq)
                query_y.append(c)

            test_query_pids = random.sample([pid for pid in pids if pid not in used_patients], 5)
            used_patients.update(test_query_pids)

            tq_files = []
            for tqpid in test_query_pids:
                tq_files.extend(self.class_to_patient_files[c][tqpid])
            chosen_tq_files = tq_files[:self.test_q_query]

            for tqf in chosen_tq_files:
                Xtq = self._load_and_preprocess(os.path.join(self.root, tqf))
                test_query_x.append(Xtq)
                test_query_y.append(c)

        query_x = torch.stack(query_x)
        query_y = torch.LongTensor(query_y)
        test_query_x = torch.stack(test_query_x)
        test_query_y = torch.LongTensor(test_query_y)

        return (query_x, query_y), (test_query_x, test_query_y)

    def __getitem__(self, index):
        """Return a new support set while keeping query and test query sets fixed."""
        support_x, support_y = [], []
    
        for c in self.classes:
            support_pid = self.predefined_support_patients[c][index % len(self.predefined_support_patients[c])]
            s_files = self.class_to_patient_files[c][support_pid][:self.k_shot]
    
            for sf in s_files:
                Xs = self._load_and_preprocess(os.path.join(self.root, sf))
                support_x.append(Xs)
                support_y.append(c)
    
        support_x = torch.stack(support_x)
        support_y = torch.LongTensor(support_y)
    
        query_x, query_y = self.fixed_query
        test_query_x, test_query_y = self.fixed_test_query
    
        return support_x, support_y, query_x, query_y, test_query_x, test_query_y

    def _get_label(self, file_path):
        with h5py.File(file_path, "r") as hdf:
            return int(hdf['label'][0]) - 1

    def _get_patient_id(self, filename):
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        return parts[0] if parts[0].isdigit() else parts[1]

    def _load_and_preprocess(self, file_path):
        with h5py.File(file_path, "r") as hdf:
            X = hdf['signal'][()]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8)
        return torch.FloatTensor(X)


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


  


def train_one_episode(
    support_x, support_y,
    query_x, query_y,
    model_biot, classifier, optimizer, criterion,
    device, n_way, num_epochs, patience,
    support_variation_index, total_variations
):
    """
    Train classifier for one variation (one support-query set pairing),
    with early stopping on query loss.

    Returns
    -------
    classifier : torch.nn.Module
        Trained classifier.
    query_accuracies : list
        A list containing the query accuracy at each epoch.
    """
    best_query_loss = float('inf')
    early_stop_counter = 0
    query_accuracies = []

    with tqdm(total=num_epochs, desc=f"Support {support_variation_index}/{total_variations}") as pbar:
        for epoch in range(num_epochs):
            classifier.train()

            # Remap and move labels/tensors to device
            support_y_mapped = remap_labels(support_y).to(device)
            query_y_mapped = remap_labels(query_y).to(device)
            support_x_gpu = support_x.to(device)
            query_x_gpu = query_x.to(device)

            # Obtain embeddings from the frozen model
            with torch.no_grad():
                support_emb = model_biot(support_x_gpu.squeeze(0))
                query_emb = model_biot(query_x_gpu.squeeze(0))

            # Forward on support set
            optimizer.zero_grad()
            logits_support = classifier(support_emb)
            loss_support = criterion(logits_support, support_y_mapped.squeeze(0))
            loss_support.backward()
            optimizer.step()

            # Evaluate on query set
            classifier.eval()
            with torch.no_grad():
                logits_query = classifier(query_emb)
                loss_query = criterion(logits_query, query_y_mapped.squeeze(0))

                # Compute query accuracy for logging
                _, query_preds = torch.max(logits_query, dim=1)
                query_acc = accuracy_score(query_y_mapped.squeeze(0).cpu().numpy(),
                                           query_preds.cpu().numpy())
                query_accuracies.append(query_acc)

            # Early stopping
            if loss_query.item() < best_query_loss:
                best_query_loss = loss_query.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

            pbar.set_postfix({
                "Epoch": epoch + 1,
                "Loss (support)": f"{loss_support.item():.4f}",
                "Loss (query)": f"{loss_query.item():.4f}",
                "Query Acc": f"{query_acc:.4f}"
            })
            pbar.update(1)

    return classifier, query_accuracies



def train_model(
    loader, model_biot, classifier_fn, optimizer_fn, criterion, logger,
    device="cpu", n_way=3, num_epochs=15, save_dir="results", patience=3
):
    """
    Train model with per-variation
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Starting training procedure with support set variation...")

    # Lists to store per-variation results
    test_metrics_list = []               
    per_class_accuracies_list = []       # each entry: array of shape (n_way,) for 'accuracy'
    per_class_metrics_list = []          # each entry: dict with "precision", "recall", "f1", "average_precision" arrays

    query_accuracies_all = {}  # (Optional) for plotting query accuracy per epoch

    
    for support_variation, batch in enumerate(loader, start=1):
        logger.info(f"Processing support patient variation {support_variation}/{len(loader)}")

        classifier = classifier_fn().to(device)
        optimizer = optimizer_fn(classifier)

        # Unpack batch
        support_x, support_y, query_x, query_y, test_query_x, test_query_y = batch

        # 1) Train on support/query sets
        classifier, query_accuracies = train_one_episode(
            support_x, support_y, query_x, query_y,
            model_biot, classifier, optimizer, criterion,
            device, n_way, num_epochs, patience,
            support_variation, len(loader)
        )

        
        (argmax_metrics, argmax_per_class, test_loss, 
         test_y_true, test_y_pred, test_probs) = evaluate_on_test(
            classifier, model_biot,
            test_query_x, test_query_y,
            device, n_way, criterion
        )

        final_metrics_dict, final_per_class = compute_metrics(
            test_y_true,test_y_pred, test_probs, n_way
        )

        test_metrics_list.append(final_metrics_dict)


    def valid_vals(key):
        return [m[key] for m in test_metrics_list if m[key] is not None]

    aggregated_test_metrics = {}
    aggregated_test_metrics_std = {}

    if len(test_metrics_list) > 0:
        # e.g. keys in final_metrics_dict: "accuracy", "precision", "recall", "f1", ...
        for key in test_metrics_list[0].keys():
            vals = valid_vals(key)
            if len(vals) > 0:
                aggregated_test_metrics[key] = np.mean(vals)
                aggregated_test_metrics_std[key] = np.std(vals)
            else:
                aggregated_test_metrics[key] = None
                aggregated_test_metrics_std[key] = None
    else:
        logger.warning("No test metrics collected! Defaulting to empty.")
        aggregated_test_metrics = {}
        aggregated_test_metrics_std = {}



    # The rest of your code that logs or returns `aggregated_test_metrics`, 
    # `aggregated_test_metrics_std`, and `per_class_df` remains the same.

    return aggregated_test_metrics, aggregated_test_metrics_std


def run_varying_shots(
    model_name, loader, model_biot, device,
    num_epochs=15, patience=5, save_dir="results/results_fcnn"
):
    """
    Runs training and evaluation for a given model across varying k-shot scenarios.
    
    Parameters
    ----------
    model_name : str
        Name of the model to be evaluated (e.g., "fcnn", "transformer").
    loader : DataLoader
        DataLoader instance for feeding data into the model.
    model_biot : Model
        The base model used for feature extraction.
    device : torch.device
        The computing device (CPU or GPU) where the model will run.
    num_epochs : int, optional
        Number of training epochs (default is 15).
    patience : int, optional
        Early stopping patience to prevent overfitting (default is 5).
    save_dir : str, optional
        Directory to save results, logs, and model checkpoints (default is "results").
    
    Returns
    -------
    results : dict
        Dictionary containing aggregated metrics, standard deviations, and per-class performance for each k-shot scenario.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    logging.basicConfig(
        filename=os.path.join(save_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    logger.info(f"Starting evaluations for model: {model_name.upper()}")

    results[model_name] = {}

    for k_shot in range(1, 6):
        logger.info(f"Evaluating {model_name.upper()} with k_shot={k_shot}")
        print(f"Evaluating {model_name.upper()} with k_shot={k_shot}")

        # Reinitialize the dataset for this k_shot
        dataset = FewShotDataset(
            root="edf/few_shot_eval_3",
            files=[f for f in os.listdir("edf/few_shot_eval_3") if f.endswith(".h5")],
            n_way=3,
            k_shot=k_shot,
            q_query=30,
            test_q_query=10,
            num_episodes=5
        )
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        if model_name == "fcnn":
            classifier_fn = FCNet(input_dim=256, hidden_dim=128, output_dim=3)
        elif model_name == "transformer":
            classifier_fn = TransformerClassifier(emb_size=256, num_layers=3, nhead=1, dim_feedforward=128, n_classes=3)
            

        aggregated_metrics, aggregated_metrics_std = train_model(
            loader=loader,
            model_biot=model_biot,
            classifier_fn=lambda: classifier_fn,
            optimizer_fn=lambda model: torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3),
            criterion=torch.nn.CrossEntropyLoss(),
            logger=logger,
            device=device,
            n_way=3,
            num_epochs=num_epochs,
            patience=patience,
        )

        results[model_name][k_shot] = [aggregated_metrics, aggregated_metrics_std]

    # Optionally generate final report or comparison plots, same as your old approach
    generate_report_and_plots(results, save_dir)
    return results

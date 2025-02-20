import logging
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import faiss
from tqdm import tqdm
from utils.utils_train import FewShotDataset, remap_labels
from utils.utils_eval import compute_metrics


def faiss_fewshot_eval_episodewise(loader, model_biot, device, n_way=3, k=3):
    """
    Perform FAISS-based few-shot evaluation per episode using k-NN classification.

    Returns:
    - List of dictionaries containing classification metrics per episode.
    """
    episode_metrics = []

    for episode_idx, (support_x, support_y, query_x, query_y, test_query_x, test_query_y) in enumerate(loader, start=1):
        support_x, query_x, test_query_x = support_x.to(device), query_x.to(device), test_query_x.to(device)
        
        support_y = remap_labels(support_y.squeeze(0)).cpu().numpy()
        query_y = remap_labels(query_y.squeeze(0)).cpu().numpy()
        test_query_y = remap_labels(test_query_y.squeeze(0)).cpu().numpy()

        # Generate embeddings
        with torch.no_grad():
            support_emb = model_biot(support_x.squeeze(0)).cpu().numpy()
            test_query_emb = model_biot(test_query_x.squeeze(0)).cpu().numpy()

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(support_emb)
        faiss.normalize_L2(test_query_emb)

        # Build FAISS index
        index = faiss.IndexFlatIP(support_emb.shape[1])  
        index.add(support_emb)

        # Perform k-NN search for test query set
        test_indices = index.search(test_query_emb, k)[1][:, 0]  
        test_preds = support_y[test_indices]  

        # Compute evaluation metrics
        metrics_dict, per_class = compute_metrics(
            y_true=test_query_y, y_pred=test_preds, y_prob=np.eye(n_way)[test_preds], n_way=n_way
        )

        episode_metrics.append({"episode": episode_idx, "metrics": metrics_dict, "per_class": per_class})

    return episode_metrics


def run_faiss_varying_shots(model_biot, device, n_way=3, k_neigh=3, shots=(1,2,3,4,5), num_episodes=5, q_query=30, test_q_query=10, save_dir="results/results_faiss"):
    """
    Run FAISS evaluation for varying k_shot values.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    log_path = os.path.join(save_dir, "faiss_eval.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.info("Starting FAISS evaluations...")

    overall_results, all_per_class_metrics = [], []

    for k_shot in shots:
        logger.info(f"Evaluating FAISS with k_shot={k_shot}")
        print(f"Evaluating FAISS with k_shot={k_shot}")

        dataset = FewShotDataset(
            root="edf/few_shot_eval_3",
            files=[f for f in os.listdir("edf/few_shot_eval_3") if f.endswith(".h5")],
            n_way=n_way, k_shot=k_shot, q_query=q_query, test_q_query=test_q_query, num_episodes=num_episodes
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        episode_metrics = faiss_fewshot_eval_episodewise(loader, model_biot, device, n_way=n_way, k=k_neigh)

        # Aggregate overall metrics
        df_metrics = pd.DataFrame([ep["metrics"] for ep in episode_metrics])
        mean_vals, std_vals = df_metrics.mean().round(3).to_dict(), df_metrics.std().round(3).to_dict()
        logger.info(f"Results for k_shot={k_shot}: mean={mean_vals}, std={std_vals}")
        print(f"k_shot={k_shot}: mean={mean_vals}, std={std_vals}")

        overall_results.append({"k_shot": k_shot, "mean_metrics": mean_vals, "std_metrics": std_vals})


    # Save summary results
    df_summary = pd.DataFrame([
        {"k_shot": res["k_shot"], **{key: f"{res['mean_metrics'][key]} ± {res['std_metrics'][key]}" for key in res["mean_metrics"]}}
        for res in overall_results
    ])
    df_summary.to_csv(os.path.join(save_dir, "faiss_eval_summary.csv"), index=False)
    print("\nFinal aggregated results:\n", df_summary)

     # Convert accuracy column from "mean ± std" format to separate numeric columns
    df_summary["accuracy_mean"] = df_summary["accuracy"].astype(str).str.split(" ± ").str[0].astype(float)
    df_summary["accuracy_std"] = df_summary["accuracy"].astype(str).str.split(" ± ").str[1].astype(float)

    

    # Plot accuracy vs. k_shot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        df_summary["k_shot"],
        df_summary["accuracy_mean"],
        yerr=df_summary["accuracy_std"],
        fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2,
        label="FAISS Accuracy"
    )
    plt.title("FAISS Accuracy vs k_shot")
    plt.xlabel("Number of Shots (k_shot)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "faiss_accuracy_across_kshot.png"))
    plt.close()

    logger.info("All FAISS evaluations complete.")
    return overall_results


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Clear log
    with open("results/results_faiss/training.log", "w") as log_file:
        log_file.write("")

    # Load pre-trained BIOTEncoder
    from model import BIOTEncoder
    model_biot = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18)
    model_biot.load_state_dict(torch.load("pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"))
    model_biot.eval().to(device)

    # Run FAISS evaluation
    results = run_faiss_varying_shots(
        model_biot=model_biot,
        device=device,
        n_way=3,
        k_neigh=3,
        shots=[1, 2, 3, 4, 5],
        num_episodes=5,
        q_query=30,
        test_q_query=10,
        save_dir="results/results_faiss"
    )

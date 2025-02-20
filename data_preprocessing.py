import argparse
import os
import numpy as np
from utils.utils_data import load_up_objects

def process_edf_data(root_dir):
    train_dir = os.path.join(root_dir, "train")
    eval_dir = os.path.join(root_dir, "eval")
    
    if os.path.basename(root_dir) == "edf" and os.path.exists(train_dir) and os.path.exists(eval_dir):
        train_out_dir = os.path.join(root_dir, "processed_train")
        eval_out_dir = os.path.join(root_dir, "processed_eval")
    else:
        os.makedirs("edf", exist_ok=True)
        root = "edf"
        train_dir = os.path.join(root, "train")
        eval_dir = os.path.join(root, "eval")
        train_out_dir = os.path.join(root, "processed_train")
        eval_out_dir = os.path.join(root, "processed_eval")
    
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(eval_out_dir, exist_ok=True)
    
    fs = 250  # Sampling frequency
    
    # Process training data
    train_features = np.empty((0, 16, fs))
    train_labels = np.empty([0, 1])
    train_offending_channel = np.empty([0, 1])
    load_up_objects(
        train_dir, train_features, train_labels, train_offending_channel, train_out_dir
    )
    
    # Process evaluation data
    eval_features = np.empty((0, 16, fs))
    eval_labels = np.empty([0, 1])
    eval_offending_channel = np.empty([0, 1])
    load_up_objects(
        eval_dir, eval_features, eval_labels, eval_offending_channel, eval_out_dir
    )
    
    print("Processing complete. Data saved in:")
    print(f"- {train_out_dir}")
    print(f"- {eval_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EDF files and store in processed_train and processed_eval folders.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing train and eval folders.")
    args = parser.parse_args()
    process_edf_data(args.root_dir)
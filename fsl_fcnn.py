import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from model import BIOTEncoder
from tqdm.notebook import tqdm
from utils.utils_train import run_varying_shots

import pandas as pd


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Clear the log file
    with open("training.log", "w") as log_file:
        log_file.write("")

    # Suppose you have a pretrained BIOTEncoder
    model_biot = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18)
    model_biot.load_state_dict(torch.load("pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"))
    model_biot.eval().to(device)

    # Run varying shots
    results = run_varying_shots(
        model_name="fcnn",
        model_biot=model_biot,
        loader=None,  # The loader is reinitialized inside run_varying_shots
        device=device,
        num_epochs=500,
        patience=5,
        save_dir="results/results_fcnn"
    )



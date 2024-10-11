# %%

import torch

from spd.experiments.resid_linear.models import ResidualLinearSPDFullRankModel
from spd.utils import set_seed

# %%

if __name__ == "__main__":
    # Set up device and seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    set_seed(0)  # You can change this seed if needed

    wandb_path = "spd-resid-linear/runs/v7ajkx3j"
    # local_path = "spd/experiments/resid_linear/out/fr_seed0_topk1.10e+00_topkrecon1.00e+00_topkl2_1.00e-02_lr1.00e-02_bs1024_ft5_lay1_resid5_mlp5/model_10000.pth"

    # Load the pretrained SPD model
    model, config, label_coeffs = ResidualLinearSPDFullRankModel.from_wandb(wandb_path)
    # model, config, label_coeffs = ResidualLinearSPDFullRankModel.from_local_path(local_path)
    model.to(device)

    # Print some basic information about the model
    print(f"Model structure:\n{model}")
    print(f"Number of features: {model.n_features}")
    print(f"Embedding dimension: {model.d_embed}")
    print(f"MLP dimension: {model.d_mlp}")
    print(f"Number of layers: {model.n_layers}")
    print(f"Number of subnetworks (k): {model.k}")

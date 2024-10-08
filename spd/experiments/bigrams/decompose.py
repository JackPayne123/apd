import json
from functools import partial
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.bigrams.model import BigramDataset, BigramModel
from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.plotting import (
    plot_components,
    plot_components_fullrank,
    plot_model_functions,
    plot_piecewise_network,
    plot_subnetwork_attributions_statistics,
    plot_subnetwork_correlations,
)
from spd.experiments.piecewise.trig_functions import generate_trig_functions
from spd.func_spd import optimize
from spd.log import logger
from spd.run_spd import Config, MinimalTaskConfig, PiecewiseConfig, calc_recon_mse
from spd.utils import (
    BatchedDataLoader,
    init_wandb,
    load_config,
    save_config_to_wandb,
    set_seed,
)


def main():
    # Parameters

    A_vocab_size = 100  # A ranges from 0 to 99
    B_vocab_size = 5  # B ranges from 0 to 4
    embedding_dim = 20
    hidden_dim = 50
    batch_size = 1024

    dataset = BigramDataset(A_vocab_size, B_vocab_size)
    new_model = BigramModel(dataset.n_A, dataset.n_B, embedding_dim, hidden_dim)
    new_model.load_state_dict(torch.load("bigram_model.pt"))
    # Evaluate model
    batch_size = 12
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = Config(
        wandb_project="spd-bigrams",
        wandb_run_name="decompose",
        wandb_run_name_prefix="bigrams",
        full_rank=True,
        seed=0,
        topk=1,
        batch_topk=False,
        steps=1000,
        print_freq=100,
        lr=0.005,
        task_config=MinimalTaskConfig(k=B_vocab_size),
        batch_size=batch_size,
        topk_param_attrib_coeff=0.0,
        orthog_coeff=0.0,
        out_recon_coeff=None,
        param_match_coeff=0.0,
        topk_recon_coeff=0.0,
        topk_l2_coeff=0.0,
        lp_sparsity_coeff=None,
        pnorm=None,
        pnorm_end=None,
        lr_schedule="constant",
        sparsity_loss_type="jacobian",
        sparsity_warmup_pct=0.0,
        unit_norm_matrices=False,
        ablation_attributions=False,
        initialize_spd="xavier",
    )

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, None)
        save_config_to_wandb(config)

    set_seed(config.seed)
    logger.info(config)

    run_name = "stefan_test_"
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    optimize(
        model=None,
        config=config,
        device="cpu",
        dataloader=dataloader,
        pretrained_model=new_model,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

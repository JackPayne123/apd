import os
import random
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST, KMNIST, MNIST, FashionMNIST, VisionDataset
from tqdm import tqdm

from spd.experiments.multitask.multitask import MultiTaskDataset, MultiTaskModel
from spd.experiments.multitask.single import (
    E10MNIST,
    GenericMNISTModel,
    MultiMNISTDataset,
    MultiMNISTDatasetLoss,
    transform,
)
from spd.experiments.multitask.trivial_combine import CombinedMNISTModel
from spd.func_spd import optimize
from spd.log import logger
from spd.run_spd import Config, MinimalTaskConfig
from spd.utils import (
    init_wandb,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def main():
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []
    for name, fname in [
        ("mnist", "MNIST.pth"),
        ("kmnist", "KMNIST.pth"),
        ("fashion", "FashionMNIST.pth"),
        ("e10mnist", "E10MNIST.pth"),
    ]:
        print(f"Loading {name} model")
        model = GenericMNISTModel(input_size=28**2, hidden_size=512, num_classes=10)
        model.load_state_dict(
            torch.load(
                f"models/{name}/{fname}",
                weights_only=False,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
        )
        models.append(model)
    combined_model = CombinedMNISTModel(models)

    data_dir = "/data/apollo/torch_datasets/"
    # Load the dataset and splits
    kwargs = {"root": data_dir, "download": True, "transform": transform, "train": True}
    datasets = [
        MNIST(**kwargs),
        KMNIST(**kwargs),
        FashionMNIST(**kwargs),
        E10MNIST(**kwargs),
    ]
    dataset = MultiMNISTDataset(datasets, p=0.25)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = Config(
        wandb_project="spd-multitask",
        wandb_run_name="k=4",
        task_config=MinimalTaskConfig(k=4),
        wandb_run_name_prefix="multitask_trivial_combine",
        full_rank=True,
        seed=1,
        topk=1,
        batch_topk=False,
        steps=10_000,
        print_freq=20,
        image_freq=200,
        save_freq=1000,
        lr=5e-4,
        batch_size=batch_size,
        topk_param_attrib_coeff=1e0,
        param_match_coeff=1e3,
        topk_recon_coeff=1,
        topk_l2_coeff=1e3,
        lp_sparsity_coeff=None,
        orthog_coeff=None,
        out_recon_coeff=None,
        slow_images=True,
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

    run_name = config.wandb_run_name
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    loss = MultiMNISTDatasetLoss(n_tasks=4, n_classes=10)

    def loss_fn(
        target: Float[Tensor, "batch (n_tasks n_classes)"],
        input: Float[Tensor, "batch (n_tasks n_classes)"],
    ) -> Float[Tensor, ""]:
        return loss.comp_kl_logits(pred_logits=input, target_logits=target)

    optimize(
        model=None,
        config=config,
        device=device,
        dataloader=dataloader,
        pretrained_model=combined_model,
        out_dir=Path("out/"),
        loss_fn=loss_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

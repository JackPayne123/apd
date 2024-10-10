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
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.experiments.multitask.multitask import MultiTaskDataset, MultiTaskModel
from spd.func_spd import optimize
from spd.log import logger
from spd.run_spd import Config, MinimalTaskConfig
from spd.utils import (
    init_wandb,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def model_comparison_fn(
    target: Float[Tensor, "batch n_tasks n_classes"],
    input: Float[Tensor, "batch n_tasks n_classes"],
) -> Float[Tensor, ""]:
    if target.ndim == 2:
        n_tasks, n_classes = target.shape
    else:
        batch_size, n_tasks, n_classes = target.shape
    assert input.shape == target.shape
    assert n_tasks == 4
    assert n_classes == 10
    ce_loss = -(target * torch.log(input)).sum(dim=-1).mean()
    print(target.shape, input.shape)
    print(torch.any(torch.isnan(target)), torch.any(torch.isnan(input)))
    return ce_loss
    # loss = torch.tensor(0.0, device=full_model_out.device)
    # for i in range(n_tasks):
    #     loss = (
    #         loss
    #         + (full_model_out[..., i, :] * torch.log(k_model_out[..., i, :])).sum(dim=-1).mean()
    #     )
    # return loss


def main():
    tasks = ["mnist", "fashionmnist", "kmnist", "notmnist"]
    p_active = 0.5
    hidden_size = 512
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Prepare datasets and input sizes
    datasets_list = []
    input_sizes = []
    num_classes = []
    for task in tasks:
        if task == "mnist":
            dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            input_size = 28 * 28
            n_classes = 10
        elif task == "fashionmnist":
            dataset = datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform
            )
            input_size = 28 * 28
            n_classes = 10
        elif task == "kmnist":
            dataset = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
            input_size = 28 * 28
            n_classes = 10
        elif task == "notmnist":
            from notmnist import NotMNISTDataset

            dataset = NotMNISTDataset(root="./data/notMNIST_small")
            input_size = 28 * 28
            n_classes = 10
        elif task == "uciletters":
            from uciletters import OpticalLettersDataset

            dataset = OpticalLettersDataset(file_path="./data/letter-recognition.data")
            input_size = 16
            n_classes = 26
        else:
            raise ValueError(f"Unknown task: {task}")
        datasets_list.append(dataset)
        input_sizes.append(input_size)
        num_classes.append(n_classes)

    # Create the MultiTaskDataset
    multi_task_dataset = MultiTaskDataset(
        datasets=datasets_list, input_sizes=input_sizes, num_classes=num_classes, p_active=p_active
    )

    # DataLoader
    dataloader = DataLoader(multi_task_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model
    model = MultiTaskModel(
        input_sizes=input_sizes, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        torch.load(
            "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/multitask/models/multitask_model.pth",
            weights_only=True,
        )
    )

    config = Config(
        wandb_project="spd-multitask",
        wandb_run_name="k=5",
        task_config=MinimalTaskConfig(k=5),
        wandb_run_name_prefix="multitask",
        full_rank=True,
        seed=1,
        topk=1,
        batch_topk=False,
        steps=10_000,
        print_freq=100,
        image_freq=1000,
        save_freq=1000,
        lr=0.005,
        batch_size=batch_size,
        topk_param_attrib_coeff=1e0,
        param_match_coeff=1e2,
        topk_recon_coeff=1,
        topk_l2_coeff=1,
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

    optimize(
        model=None,
        config=config,
        device=device,
        dataloader=dataloader,
        pretrained_model=model,
        out_dir=Path("out/"),
        loss_fn=model_comparison_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

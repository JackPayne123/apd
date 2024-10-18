from collections.abc import Callable
from pathlib import Path

import einops
import fire
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST, MNIST, FashionMNIST

from spd.experiments.multitask.single import (
    E10MNIST,
    GenericMNISTModel,
    MultiMNISTDataset,
    MultiMNISTDatasetLoss,
    transform,
)
from spd.experiments.multitask.trivial_combine import CombinedMNISTModel
from spd.func_spd import functional_call, grad, optimize, plot_matrix
from spd.log import logger
from spd.run_spd import Config, MinimalTaskConfig
from spd.utils import (
    set_seed,
)

batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

models = []
model_path = Path("/data/apollo/spd/multitask_models/2024_10_16B")
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
            model_path / name / fname,
            weights_only=False,
            map_location=torch.device("cpu"),
        )
    )
    models.append(model)
pretrained_model = CombinedMNISTModel(models)

data_dir = "/data/apollo/torch_datasets/"
# Load the dataset and splits
kwargs = {"root": data_dir, "download": True, "transform": transform, "train": True}
datasets = [
    MNIST(**kwargs),
    KMNIST(**kwargs),
    FashionMNIST(**kwargs),
    E10MNIST(**kwargs),
]
dataset = MultiMNISTDataset(datasets, n=1, p=None)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

config = Config(
    wandb_project="spd-multitask",
    wandb_run_name="k=4_modelsB",
    task_config=MinimalTaskConfig(k=4),
    wandb_run_name_prefix="multitask_trivial_combine",
    full_rank=True,
    seed=1,
    topk=1,
    batch_topk=False,
    steps=20_000,
    print_freq=20,
    image_freq=200,
    save_freq=1000,
    lr=5e-4,
    batch_size=batch_size,
    topk_param_attrib_coeff=0.0,
    param_match_coeff=1e3,
    topk_recon_coeff=1,
    topk_l2_coeff=1e3,
    lp_sparsity_coeff=None,
    orthog_coeff=None,
    out_recon_coeff=None,
    slow_images=True,
    pnorm=None,
    pnorm_end=None,
    lr_schedule="cosine",
    sparsity_loss_type="jacobian",
    sparsity_warmup_pct=0.0,
    unit_norm_matrices=False,
    ablation_attributions=False,
    initialize_spd="xavier",
)

set_seed(config.seed)
logger.info(config)


device = "cpu" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

Loss = MultiMNISTDatasetLoss(n_tasks=4, n_classes=10)


def loss_fn(
    target: Float[Tensor, "batch (n_tasks n_classes)"],
    input: Float[Tensor, "batch (n_tasks n_classes)"],
) -> Float[Tensor, ""]:
    return Loss.comp_kl_logits(pred_logits=input, target_logits=target)


out_dir = (Path("out/"),)

k = config.task_config.k


def model_func(
    single_mask: Float[Tensor, " k"],
    single_batch: Float[Tensor, " n_inputs"],
    k_params: dict[str, Float[Tensor, " k ..."]],
) -> Float[Tensor, " n_outputs"]:
    summed_params = {
        k: einops.einsum(v, single_mask, "k ..., k -> ...") for k, v in k_params.items()
    }
    return functional_call(pretrained_model, summed_params, single_batch)


result = torch.zeros(k, k)

k_params = torch.load("out/k_params_step_20000.pt", map_location=torch.device("cpu"))
assert k is not None
for task_idx in range(k):
    task_mask = torch.zeros(k, device="cpu", dtype=torch.bool)
    task_mask[task_idx] = 1
    dataset.mask = task_mask
    for k_idx in range(k):
        single_batch, target = dataset[0]
        single_batch.to(device)
        # print("Input", single_batch.reshape(4, -1).sum(dim=-1))
        pretrained_out = pretrained_model(single_batch)
        # print("Pretrained", pretrained_out.reshape(4, -1).sum(dim=-1))
        single_batch = single_batch.to(device)
        model_mask = torch.zeros(k, device=device)
        model_mask[k_idx] = 1.0
        spd_out = model_func(model_mask, single_batch, k_params)
        spd_out.to(device)
        pretrained_out.to(device)
        # print("SPD", spd_out.reshape(4, -1).sum(dim=-1))
        loss = loss_fn(pretrained_out, spd_out)
        print(task_idx, k_idx, loss)
        result[task_idx, k_idx] = loss

import matplotlib.pyplot as plt

plt.imshow(result.cpu().detach().numpy(), cmap="RdBu", vmin=-0.02, vmax=0.02)
plt.colorbar()

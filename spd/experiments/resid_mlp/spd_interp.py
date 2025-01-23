# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from pydantic import PositiveFloat
from torch import Tensor
from tqdm import tqdm

from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDRankPenaltyModel
from spd.experiments.resid_mlp.plotting import (
    analyze_per_feature_performance,
    collect_average_components_per_feature,
    collect_per_feature_losses,
    get_feature_subnet_map,
    plot_feature_response_with_subnets,
    plot_per_feature_performance,
    plot_spd_feature_contributions_truncated,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.experiments.resid_mlp.resid_mlp_decomposition import plot_subnet_categories
from spd.run_spd import ResidualMLPTaskConfig
from spd.settings import REPO_ROOT
from spd.utils import (
    COLOR_PALETTE,
    SPDOutputs,
    run_spd_forward_pass,
    set_seed,
)

color_map = {
    "target": COLOR_PALETTE[0],
    "apd_topk": COLOR_PALETTE[1],
    "apd_scrubbed": COLOR_PALETTE[4],
    "apd_antiscrubbed": COLOR_PALETTE[2],  # alt: 3
    "baseline_monosemantic": "grey",
}

out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out/figures/"
out_dir.mkdir(parents=True, exist_ok=True)

# %% Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed

use_data_from_files = True
wandb_path = (
    "wandb:spd-resid-mlp/runs/8qz1si1l"  # 1 layer (40k steps. 15 cross 98 mono) R6 in paper
)
# wandb_path = "wandb:spd-resid-mlp/runs/9a639c6w"  # 1 layer topk=1
# wandb_path = "wandb:spd-resid-mlp/runs/cb0ej7hj"  # 2 layer 2LR4 in paper
# wandb_path = "wandb:spd-resid-mlp/runs/wbeghftm"  # 2 layer topk=1
# wandb_path = "wandb:spd-resid-mlp/runs/c1q3bs6f"  # 2 layer m=1

wandb_id = wandb_path.split("/")[-1]

# Load the pretrained SPD model
model, config, label_coeffs = ResidualMLPSPDRankPenaltyModel.from_pretrained(wandb_path)
assert isinstance(config.task_config, ResidualMLPTaskConfig)

# Path must be local
target_model, target_model_train_config_dict, target_label_coeffs = (
    ResidualMLPModel.from_pretrained(config.task_config.pretrained_model_path)
)
# Print some basic information about the model
print(f"Number of features: {model.config.n_features}")
print(f"Feature probability: {config.task_config.feature_probability}")
print(f"Embedding dimension: {model.config.d_embed}")
print(f"MLP dimension: {model.config.d_mlp}")
print(f"Number of layers: {model.config.n_layers}")
print(f"Number of subnetworks (k): {model.config.k}")
model = model.to(device)
label_coeffs = label_coeffs.to(device)
target_model = target_model.to(device)
target_label_coeffs = target_label_coeffs.to(device)
assert torch.allclose(target_label_coeffs, label_coeffs)

n_layers = target_model.config.n_layers

# %% Plot how many subnets are monosemantic, etc.
fig = plot_subnet_categories(model, device, cutoff=4e-2)
fig.show()


# %%
def spd_model_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk: PositiveFloat | None = config.topk,
    batch_topk: bool = config.batch_topk,
) -> SPDOutputs:
    assert topk is not None
    return run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        batch_topk=batch_topk,
        topk=topk,
        distil_from_target=config.distil_from_target,
    )


def target_model_fn(batch: Float[Tensor, "batch n_instances"]):
    return target_model(batch)[0]


# %%
dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=True,
    label_type=target_model_train_config_dict["label_type"],
    act_fn_name=target_model.config.act_fn_name,
    label_coeffs=target_label_coeffs,
    data_generation_type="at_least_zero_active",  # We will change this in the for loop
)

per_feature_losses_path = Path(out_dir) / f"resid_mlp_losses_{n_layers}layers_{wandb_id}.pt"
if not use_data_from_files or not per_feature_losses_path.exists():
    loss_target, loss_spd_batch_topk, loss_spd_sample_topk = collect_per_feature_losses(
        target_model=target_model,
        spd_model=model,
        config=config,
        dataset=dataset,
        device=device,
        batch_size=config.batch_size,
        n_samples=100_000,
    )
    # Save the losses to a file
    torch.save(
        (loss_target, loss_spd_batch_topk, loss_spd_sample_topk),
        per_feature_losses_path,
    )

# Load the losses from a file
loss_target, loss_spd_batch_topk, loss_spd_sample_topk = torch.load(
    per_feature_losses_path, weights_only=True, map_location="cpu"
)

# %%
# New per-feature performance plots
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
axs = np.array(axs)

indices = loss_target.argsort()
topk = int(config.topk) if config.topk is not None and config.topk == 1 else config.topk
plot_per_feature_performance(
    losses=loss_spd_batch_topk,
    sorted_indices=indices,
    ax=axs[1],
    label=f"APD (per-batch top-k={topk})",
    color=color_map["apd_topk"],
)

plot_per_feature_performance(
    losses=loss_target,
    sorted_indices=indices,
    ax=axs[1],
    label="Target model",
    color=color_map["target"],
)
axs[1].legend(loc="upper left")

plot_per_feature_performance(
    losses=loss_spd_sample_topk,
    sorted_indices=indices,
    ax=axs[0],
    label="APD (per-sample top-k=1)",
    color=color_map["apd_topk"],
)
plot_per_feature_performance(
    losses=loss_target,
    sorted_indices=indices,
    ax=axs[0],
    label="Target model",
    color=color_map["target"],
)

axs[0].legend(loc="upper left", fontsize=12)
axs[1].legend(loc="upper left", fontsize=12)

# Use the max y-axis limit for both subplots
max_ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
axs[0].set_ylim(0, max_ylim)
axs[1].set_ylim(0, max_ylim)


# Remove the top and right spines
for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# # Increase the fontsize of the xlabel and ylabel
for ax in axs:
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

fig.show()
fig.savefig(out_dir / f"resid_mlp_per_feature_performance_{n_layers}layers_{wandb_id}.png")
print(
    f"Saved figure to {out_dir / f'resid_mlp_per_feature_performance_{n_layers}layers_{wandb_id}.png'}"
)


# %%
# Scatter plot of avg active components vs loss difference
dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=True,
    label_type=target_model_train_config_dict["label_type"],
    act_fn_name=target_model.config.act_fn_name,
    label_coeffs=target_label_coeffs,
    data_generation_type="at_least_zero_active",  # We will change this in the for loop
)


avg_components_path = Path(out_dir) / f"avg_components_{n_layers}layers_{wandb_id}.pt"
if not use_data_from_files or not avg_components_path.exists():
    avg_components = collect_average_components_per_feature(
        model_fn=spd_model_fn,
        dataset=dataset,
        device=device,
        n_features=model.config.n_features,
        batch_size=config.batch_size,
        n_samples=500_000,
    )
    # Save the avg_components to a file
    torch.save(avg_components.cpu(), avg_components_path)

# Load the avg_components from a file
avg_components = torch.load(avg_components_path, map_location=device, weights_only=True)

# Get the loss of the spd model w.r.t the target model
fn_without_batch_topk = lambda batch: spd_model_fn(
    batch, topk=1, batch_topk=False
).spd_topk_model_output  # type: ignore
losses_spd_wrt_target = analyze_per_feature_performance(
    model_fn=fn_without_batch_topk,
    target_model_fn=target_model_fn,
    model_config=model.config,
    device=device,
    batch_size=config.batch_size,
)


def plot_avg_components_scatter(
    losses_spd_wrt_target: Float[Tensor, " n_features"],
    avg_components: Float[Tensor, " n_features"],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(
        losses_spd_wrt_target.abs().detach().cpu(),
        avg_components.detach().cpu(),
    )
    ax.set_xlabel("MSE between APD (per-sample top-k=1) and target model outputs")
    ax.set_ylabel("Average number of active components")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Increase the fontsize of the xlabel and ylabel
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    return fig


fig = plot_avg_components_scatter(
    losses_spd_wrt_target=losses_spd_wrt_target, avg_components=avg_components
)
fig.show()
# Save the figure
fig.savefig(out_dir / f"resid_mlp_avg_components_scatter_{n_layers}layers_{wandb_id}.png")
print(
    f"Saved figure to {out_dir / f'resid_mlp_avg_components_scatter_{n_layers}layers_{wandb_id}.png'}"
)

# %%
# Plot the main truncated feature contributions figure for the paper
fig = plot_spd_feature_contributions_truncated(
    spd_model=model,
    target_model=target_model,
    device=device,
    n_features=10,
    include_crossterms=False,
)
fig.show()
# Save the figure
out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out"
fig.savefig(out_dir / f"resid_mlp_weights_{n_layers}layers_{wandb_id}.png")
print(f"Saved figure to {out_dir / f'resid_mlp_weights_{n_layers}layers_{wandb_id}.png'}")


# %% Collect data for causal scrubbing-esque test


def top1_model_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk_mask: Float[Tensor, "batch n_instances k"] | None,
) -> SPDOutputs:
    """Top1 if topk_mask is None, else just use provided topk_mask"""
    topk_mask = topk_mask.to(device) if topk_mask is not None else None
    assert config.topk is not None
    return run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        batch_topk=False,
        topk=1,
        distil_from_target=config.distil_from_target,
        topk_mask=topk_mask,
    )


# Dictionary feature_idx -> subnet_idx
subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

batch_size = config.batch_size
# make sure to use config.batch_size because
# it's tuned to config.topk!
n_batches = 100  # 100 and 1000 are similar. Maybe use 1000 for final plots only
test_dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=False,  # Our labels will be the output of the target model
    data_generation_type="at_least_zero_active",
)

# Initialize tensors to store all losses
all_loss_scrubbed = []
all_loss_antiscrubbed = []
all_loss_random = []
all_loss_spd = []
all_loss_zero = []
all_loss_monosemantic = []
for _ in tqdm(range(n_batches)):
    # In the future this will be merged into generate_batch
    batch = dataset._generate_multi_feature_batch_no_zero_samples(batch_size, buffer_ratio=2)
    if isinstance(dataset, ResidualMLPDataset) and dataset.label_fn is not None:
        labels = dataset.label_fn(batch)
    else:
        labels = batch.clone().detach()

    batch = batch.to(device)
    active_features = torch.where(batch != 0)
    # Randomly assign 0 or 1 to topk mask
    random_topk_mask = torch.randint(0, 2, (batch_size, model.config.n_instances, model.config.k))
    scrubbed_topk_mask = torch.randint(0, 2, (batch_size, model.config.n_instances, model.config.k))
    antiscrubbed_topk_mask = torch.randint(
        0, 2, (batch_size, model.config.n_instances, model.config.k)
    )
    for b, i, f in zip(*active_features, strict=False):
        s = subnet_indices[f.item()]
        scrubbed_topk_mask[b, i, s] = 1
        antiscrubbed_topk_mask[b, i, s] = 0
    topk = config.topk
    batch_topk = config.batch_topk

    out_spd = spd_model_fn(batch, topk=topk, batch_topk=batch_topk).spd_topk_model_output
    out_random = top1_model_fn(batch, random_topk_mask).spd_topk_model_output
    out_scrubbed = top1_model_fn(batch, scrubbed_topk_mask).spd_topk_model_output
    out_antiscrubbed = top1_model_fn(batch, antiscrubbed_topk_mask).spd_topk_model_output
    out_target = target_model_fn(batch)
    # Monosemantic baseline
    out_monosemantic = batch.clone()
    d_mlp = target_model.config.d_mlp * target_model.config.n_layers  # type: ignore
    out_monosemantic[..., :d_mlp] = labels[..., :d_mlp]

    # Calc MSE losses
    all_loss_scrubbed.append(
        ((out_scrubbed - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
    )
    all_loss_antiscrubbed.append(
        ((out_antiscrubbed - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
    )
    all_loss_random.append(((out_random - out_target) ** 2).mean(dim=-1).flatten().detach().cpu())
    all_loss_spd.append(((out_spd - out_target) ** 2).mean(dim=-1).flatten().detach().cpu())
    all_loss_zero.append(
        ((torch.zeros_like(out_target) - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
    )
    all_loss_monosemantic.append(
        ((out_monosemantic - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
    )

# Concatenate all batches
loss_scrubbed = torch.cat(all_loss_scrubbed)
loss_antiscrubbed = torch.cat(all_loss_antiscrubbed)
loss_random = torch.cat(all_loss_random)
loss_spd = torch.cat(all_loss_spd)
loss_zero = torch.cat(all_loss_zero)
loss_monosemantic = torch.cat(all_loss_monosemantic)

print(f"Loss SPD:           {loss_spd.mean().item():.6f}")
print(f"Loss scrubbed:      {loss_scrubbed.mean().item():.6f}")
print(f"Loss antiscrubbed:  {loss_antiscrubbed.mean().item():.6f}")
print(f"Loss monosemantic:  {loss_monosemantic.mean().item():.6f}")
print(f"Loss random:        {loss_random.mean().item():.6f}")
print(f"Loss zero:          {loss_zero.mean().item():.6f}")

# %%
# Plot causal scrubbing-esque test

# TODO orange bump: maybe SPD is using 2 subnets for some
# features? Would explain scrubbed and antiscrubbed bimodality,
# and random (which is just scrubbed + antiscrubbed) too.

fig, ax = plt.subplots(figsize=(15, 5))
log_bins: list[float] = np.geomspace(1e-7, loss_zero.max().item(), 50).tolist()  # type: ignore
ax.hist(
    loss_spd,
    bins=log_bins,
    label="APD (top-k)",
    histtype="step",
    lw=2,
    color=color_map["apd_topk"],
)
ax.axvline(loss_spd.mean().item(), color=color_map["apd_topk"], linestyle="--")
ax.hist(
    loss_scrubbed,
    bins=log_bins,
    label="APD (scrubbed)",
    histtype="step",
    lw=2,
    color=color_map["apd_scrubbed"],
)
ax.axvline(loss_scrubbed.mean().item(), color=color_map["apd_scrubbed"], linestyle="--")
ax.hist(
    loss_antiscrubbed,
    bins=log_bins,
    label="APD (anti-scrubbed)",
    histtype="step",
    lw=2,
    color=color_map["apd_antiscrubbed"],
)
ax.axvline(loss_antiscrubbed.mean().item(), color=color_map["apd_antiscrubbed"], linestyle="--")
# ax.hist(loss_random, bins=log_bins, label="APD (random)", histtype="step")
# ax.hist(loss_zero, bins=log_bins, label="APD (zero)", histtype="step")
ax.axvline(
    loss_monosemantic.mean().item(),
    color=color_map["baseline_monosemantic"],
    linestyle="-",
    label="Monosemantic neuron solution",
)
ax.legend()
ax.set_ylabel(f"Count (out of {batch_size * n_batches} samples)")
ax.set_xlabel("MSE loss with target model output")
ax.set_xscale("log")

# Remove spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# fig.suptitle("Losses when scrubbing set of parameter components")
fig.savefig(
    out_dir / f"resid_mlp_scrub_hist_{n_layers}layers_{wandb_id}.png", bbox_inches="tight", dpi=300
)
print(f"Saved figure to {out_dir / f'resid_mlp_scrub_hist_{n_layers}layers_{wandb_id}.png'}")
fig.show()


# %% Linearity test: Enable one subnet after the other
# candlestick plot

# # Dictionary feature_idx -> subnet_idx
subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

n_features = model.config.n_features
feature_idx = 42
subtract_inputs = True  # TODO TRUE subnet


fig = plot_feature_response_with_subnets(
    topk_model_fn=top1_model_fn,
    device=device,
    model_config=model.config,
    feature_idx=feature_idx,
    subnet_idx=subnet_indices[feature_idx],
    batch_size=1000,
    plot_type="errorbar",
    color_map=color_map,
)["feature_response_with_subnets"]
fig.savefig(  # type: ignore
    out_dir / f"feature_response_with_subnets_{feature_idx}_{n_layers}layers_{wandb_id}.png",
    bbox_inches="tight",
    dpi=300,
)
print(
    f"Saved figure to {out_dir / f'feature_response_with_subnets_{feature_idx}_{n_layers}layers_{wandb_id}.png'}"
)
plt.show()

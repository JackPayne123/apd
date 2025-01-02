# %% Imports


import einops
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
    get_feature_subnet_map,
    plot_feature_response_with_subnets,
    plot_individual_feature_response,
    plot_spd_feature_contributions_truncated,
    plot_spd_relu_contribution,
    plot_virtual_weights_target_spd,
    spd_calculate_virtual_weights,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.experiments.resid_mlp.resid_mlp_decomposition import plot_subnet_categories
from spd.plotting import collect_sparse_dataset_mse_losses, plot_sparse_feature_mse_line_plot
from spd.run_spd import ResidualMLPTaskConfig
from spd.settings import REPO_ROOT
from spd.utils import DataGenerationType, SPDOutputs, calc_recon_mse, run_spd_forward_pass, set_seed

# %% Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed

wandb_path = "wandb:spd-resid-mlp/runs/8qz1si1l"  # 1 layer (40k steps. 15 cross 98 mono) R6
# wandb_path = "wandb:spd-resid-mlp/runs/yk6we9kl"  # New 1 layer with 100 mono
# wandb_path = "wandb:spd-resid-mlp/runs/cb0ej7hj"  # 2 layer 2LR4
# Load the pretrained SPD model
model, config, label_coeffs = ResidualMLPSPDRankPenaltyModel.from_pretrained(wandb_path)
assert isinstance(config.task_config, ResidualMLPTaskConfig)

# Path must be local
target_model, target_model_train_config_dict, target_label_coeffs = (
    ResidualMLPModel.from_pretrained(config.task_config.pretrained_model_path)
)
model = model.to(device)
label_coeffs = label_coeffs.to(device)
target_model = target_model.to(device)
target_label_coeffs = target_label_coeffs.to(device)
assert torch.allclose(target_label_coeffs, torch.tensor(label_coeffs))

n_layers = target_model.config.n_layers
# %%
fig = plot_subnet_categories(model, device, cutoff=4e-2)
fig.show()

# %%


def spd_model_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk: PositiveFloat | None = config.topk,
    batch_topk: bool = config.batch_topk,
) -> Float[Tensor, "batch n_instances n_features"]:
    assert topk is not None
    return run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        batch_topk=batch_topk,
        topk=topk,
        distil_from_target=config.distil_from_target,
    ).spd_topk_model_output


def target_model_fn(batch: Float[Tensor, "batch n_instances"]):
    return target_model(batch)[0]


# Plot per-feature performance when setting topk=1 and batch_topk=False
fig, ax1 = plt.subplots(figsize=(15, 5))
sorted_indices = analyze_per_feature_performance(
    model_fn=target_model_fn,
    model_config=target_model.config,
    ax=ax1,
    label="Target",
    device=device,
    batch_size=config.batch_size,
    sorted_indices=None,
)
fn_without_batch_topk = lambda batch: spd_model_fn(batch, topk=1, batch_topk=False)  # type: ignore
analyze_per_feature_performance(
    model_fn=fn_without_batch_topk,
    model_config=model.config,
    ax=ax1,
    label="SPD",
    device=device,
    batch_size=config.batch_size,
    sorted_indices=sorted_indices,
)
ax1.legend()
fig.show()

# Plot per-feature performance when using batch_topk
fig, ax2 = plt.subplots(figsize=(15, 5))
sorted_indices = analyze_per_feature_performance(
    model_fn=target_model_fn,
    model_config=target_model.config,
    ax=ax2,
    label="Target",
    device=device,
    batch_size=config.batch_size,
    sorted_indices=None,
)
analyze_per_feature_performance(
    model_fn=spd_model_fn,
    model_config=model.config,
    ax=ax2,
    label="SPD",
    device=device,
    sorted_indices=sorted_indices,
    batch_size=config.batch_size,
)
ax2.legend()
# Use the same y-axis limits as the topk=1 plot
ax2.set_ylim(ax1.get_ylim())
fig.show()


# %%
# Plot the main truncated feature contributions figure for the paper
fig = plot_spd_feature_contributions_truncated(
    spd_model=model,
    target_model=target_model,
    device=device,
    n_features=10,
)
fig.show()
# Save the figure
out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out"
fig.savefig(out_dir / f"resid_mlp_weights_{n_layers}layers.png")
print(f"Saved figure to {out_dir / f'resid_mlp_weights_{n_layers}layers.png'}")
# %%
# Get the entries for the main loss table in the paper
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
gen_types: list[DataGenerationType] = [
    "at_least_zero_active",
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
]
assert config.topk is not None
results = collect_sparse_dataset_mse_losses(
    dataset=dataset,
    target_model=target_model,
    spd_model=model,
    batch_size=10000,  # Similar to 1k. Only do 10k on a gpu, slow otherwise
    device=device,
    topk=config.topk,
    attribution_type=config.attribution_type,
    batch_topk=config.batch_topk,
    distil_from_target=config.distil_from_target,
    gen_types=gen_types,
    buffer_ratio=2,
)

# Convert all results to floats
results = {
    gen_type: {k: float(v.detach().cpu()) for k, v in results[gen_type].items()}
    for gen_type in gen_types
}
# %%
# Create line plot of results

label_map = {
    "baseline_monosemantic": "Baseline target model (monosemantic neurons)",
    "target": "Target model",
    "spd": "APD model",
}

fig = plot_sparse_feature_mse_line_plot(results, label_map=label_map)
fig.show()
fig.savefig(out_dir / f"resid_mlp_mse_{n_layers}layers.png")
print(f"Saved figure to {out_dir / f'resid_mlp_mse_{n_layers}layers.png'}")

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

    out_spd = spd_model_fn(batch, topk=topk, batch_topk=batch_topk)
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
log_bins = np.geomspace(1e-7, loss_zero.max().item(), 50).tolist()
ax.hist(
    loss_spd,
    bins=log_bins,  # type: ignore
    label="APD (top-k)",
    histtype="step",
    lw=2,
    color="tab:purple",
)
ax.axvline(loss_spd.mean().item(), color="tab:purple", linestyle="--")
ax.hist(
    loss_scrubbed,
    bins=log_bins,  # type: ignore
    label="APD (scrubbed)",
    histtype="step",
    lw=2,
    color="tab:orange",
)
ax.axvline(loss_scrubbed.mean().item(), color="tab:orange", linestyle="--")
ax.hist(
    loss_antiscrubbed,
    bins=log_bins,  # type: ignore
    label="APD (anti-scrubbed)",
    histtype="step",
    lw=2,
    color="tab:green",
)
ax.axvline(loss_antiscrubbed.mean().item(), color="tab:green", linestyle="--")
# ax.hist(loss_random, bins=log_bins, label="APD (random)", histtype="step")
# ax.hist(loss_zero, bins=log_bins, label="APD (zero)", histtype="step")
ax.axvline(
    loss_monosemantic.mean().item(),
    color="black",
    linestyle="--",
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
fig.savefig(out_dir / f"resid_mlp_scrub_hist_{n_layers}layers.png", bbox_inches="tight", dpi=300)
print(f"Saved figure to {out_dir / f'resid_mlp_scrub_hist_{n_layers}layers.png'}")
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
    batch_size=10000,
    plot_type="errorbar",
)["feature_response_with_subnets"]
if fig is not None:
    fig.savefig(  # type: ignore
        out_dir / f"feature_response_with_subnets_{feature_idx}_{n_layers}layers.png",
        bbox_inches="tight",
        dpi=300,
    )
    print(
        f"Saved figure to {out_dir / f'feature_response_with_subnets_{feature_idx}_{n_layers}layers.png'}"
    )
    plt.show()


################## End of current paper plots ##################


# %%
dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=False,  # Our labels will be the output of the target model
    data_generation_type=config.task_config.data_generation_type,
)
if config.task_config.data_generation_type == "at_least_zero_active":
    # In the future this will be merged into generate_batch
    batch = dataset._generate_multi_feature_batch_no_zero_samples(config.batch_size, buffer_ratio=2)
    if isinstance(dataset, ResidualMLPDataset) and dataset.label_fn is not None:
        labels = dataset.label_fn(batch)
    else:
        labels = batch.clone().detach()
else:
    batch, labels = dataset.generate_batch(config.batch_size)
batch = batch.to(device)
labels = labels.to(device)
# Print some basic information about the model
print(f"Number of features: {model.config.n_features}")
print(f"Embedding dimension: {model.config.d_embed}")
print(f"MLP dimension: {model.config.d_mlp}")
print(f"Number of layers: {model.config.n_layers}")
print(f"Number of subnetworks (k): {model.config.k}")

target_model_output, _, _ = target_model(batch)

assert config.topk is not None
spd_outputs = run_spd_forward_pass(
    spd_model=model,
    target_model=target_model,
    input_array=batch,
    attribution_type=config.attribution_type,
    batch_topk=config.batch_topk,
    topk=config.topk,
    distil_from_target=config.distil_from_target,
)
topk_recon_loss = calc_recon_mse(
    spd_outputs.spd_topk_model_output, target_model_output, has_instance_dim=True
)
print(f"Topk recon loss: {np.array(topk_recon_loss.detach().cpu())}")

# Print param shapes for model
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")


# %% Feature-relu contribution plots

fig1, fig2 = plot_spd_relu_contribution(model, target_model, device, k_plot_limit=3)
fig1.suptitle("How much does each ReLU contribute to each feature?")
fig2.suptitle("How much does each feature route through each ReLU?")


# %% Individual feature response


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), constrained_layout=True)
axes = np.atleast_2d(axes)  # type: ignore
plot_individual_feature_response(
    model_fn=target_model_fn,
    device=device,
    model_config=model.config,
    ax=axes[0, 0],
)
plot_individual_feature_response(
    model_fn=target_model_fn,
    device=device,
    model_config=model.config,
    sweep=True,
    ax=axes[1, 0],
)

plot_individual_feature_response(
    model_fn=spd_model_fn,
    device=device,
    model_config=model.config,
    ax=axes[0, 1],
)
plot_individual_feature_response(
    model_fn=spd_model_fn,
    device=device,
    model_config=model.config,
    sweep=True,
    ax=axes[1, 1],
)
axes[0, 0].set_ylabel(axes[0, 0].get_title())
axes[1, 0].set_ylabel(axes[1, 0].get_title())
axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")
axes[0, 0].set_title("Target model")
axes[0, 1].set_title("SPD model")
axes[1, 0].set_title("")
axes[1, 1].set_title("")
axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
fig.show()

# %% Per-feature performance
fig, ax = plt.subplots(figsize=(15, 5))
sorted_indices = analyze_per_feature_performance(
    model_fn=spd_model_fn,
    model_config=model.config,
    ax=ax,
    label="SPD",
    device=device,
    sorted_indices=None,
    zorder=1,
)
analyze_per_feature_performance(
    model_fn=target_model_fn,
    model_config=target_model.config,
    ax=ax,
    label="Target",
    device=device,
    sorted_indices=sorted_indices,
    zorder=0,
)
ax.legend()
fig.show()


# %% Virtual weights
fig = plot_virtual_weights_target_spd(target_model, model, device)
fig.show()

# %% Analysis of one feature / subnetwork, picking feature 1 because it looks sketch.

# Subnet combinations relevant for feature 1
virtual_weights = spd_calculate_virtual_weights(model, device)
in_conns: Float[Tensor, "k1 n_features1 d_mlp"] = virtual_weights["in_conns"][0]
out_conns: Float[Tensor, "k2 d_mlp n_features2"] = virtual_weights["out_conns"][0]
relu_conns_sum: Float[Tensor, "k1 k2 f1 f2"] = einops.einsum(
    in_conns, out_conns, "k1 f1 d_mlp, k2 d_mlp f2 -> k1 k2 f1 f2"
)
plt.matshow(relu_conns_sum[:, :, 1, 1].detach().cpu())
plt.title("Subnet combinations relevant for feature 1")
plt.show()

# Per-neuron contribution to feature 1
relu_conns: Float[Tensor, "k1 k2 f1 f2"] = einops.einsum(
    in_conns, out_conns, "k1 f1 d_mlp, k2 d_mlp f2 -> k1 k2 f1 f2 d_mlp"
)
plt.plot(relu_conns[1, 1, 1, 1, :].detach().cpu(), label="Subnet 1 of W_in and W_out")
plt.plot(
    relu_conns[:, :, 1, 1, :].sum(dim=(0, 1)).detach().cpu(),
    label="All subnets (i,j) of W_in and W_out",
)
plt.plot(
    relu_conns[:, :, 1, 1, :].sum(dim=(0, 1)).detach().cpu()
    - relu_conns[1, 1, 1, 1, :].detach().cpu(),
    label="All subnets (i,j) != (1,1) of W_in and W_out",
)
plt.title("Per-neuron contribution to feature 1")
plt.xlabel("Neuron")
plt.ylabel("Weight")
plt.legend()
plt.show()

# Which subnets contain the neuron-45 contribution to feature 1?
plt.matshow(relu_conns[:, :, 1, 1, 45].detach().cpu())
plt.title("Which subnets contain the neuron-45 contribution to feature 1?")
print("Seems to be the diagonal k1=95, k2=95 term", relu_conns[:, :, 1, 1, 45].argmax())

# %%

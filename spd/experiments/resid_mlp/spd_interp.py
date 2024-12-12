# %% Imports


from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDRankPenaltyModel
from spd.experiments.resid_mlp.plotting import (
    analyze_per_feature_performance,
    get_feature_subnet_map,
    plot_feature_response_with_subnets,
    plot_individual_feature_response,
    plot_resid_vs_mlp_out,
    plot_spd_relu_contribution,
    plot_virtual_weights_target_spd,
    spd_calculate_virtual_weights,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import ResidualMLPTaskConfig, calc_recon_mse
from spd.utils import SPDOutputs, run_spd_forward_pass, set_seed

# %% Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed
path = "wandb:spd-resid-mlp/runs/8st8rale"  # Dan's R1
# path = "wandb:spd-resid-mlp/runs/kkstog7o"  # Dan's R2
# path = "wandb:spd-resid-mlp/runs/2ala9kjy"  # Stefan's initial try
# path = "wandb:spd-resid-mlp/runs/qmio77cl"  # Stefan's run with initial-hardcoded topk
# Load the pretrained SPD model
model, config, label_coeffs = ResidualMLPSPDRankPenaltyModel.from_pretrained(path)
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
assert torch.allclose(target_label_coeffs, torch.tensor(label_coeffs))


# %% Check performance for different numbers of active features

data_generation_types: list[
    Literal["at_least_zero_active", "exactly_one_active", "exactly_two_active"]
] = ["at_least_zero_active", "exactly_one_active", "exactly_two_active"]
for data_generation_type in data_generation_types:
    batch_size = 10_000
    dataset = ResidualMLPDataset(
        n_instances=model.config.n_instances,
        n_features=model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        data_generation_type=data_generation_type,
    )
    instance_idx = 0
    batch, labels = dataset.generate_batch(batch_size)
    batch = batch.to(device)
    labels = labels.to(device)
    target_model_output, _, _ = target_model(batch)
    assert config.topk is not None
    batch_topk = data_generation_type == "at_least_zero_active"
    print(f"Topk recon loss for {data_generation_type} (batch_topk={batch_topk}):")
    topk_recon_losses = []
    feature_subnet_correlations = []
    topks = [config.topk, 1, 2, 3]
    for topk in topks:
        spd_outputs = run_spd_forward_pass(
            spd_model=model,
            target_model=target_model,
            input_array=batch,
            attribution_type=config.attribution_type,
            batch_topk=batch_topk,
            topk=topk,
            distil_from_target=config.distil_from_target,
        )
        topk_recon_loss: Float[Tensor, " batch"] = (
            (spd_outputs.spd_topk_model_output - target_model_output) ** 2
        ).mean(dim=(-2, -1))
        topk_recon_losses.append(topk_recon_loss)
        print(f"Top-k with k={topk}: {topk_recon_loss.mean().item():.6f}")
    # Histograms
    fig, ax = plt.subplots(figsize=(15, 5))
    for i, topk_recon_loss in enumerate(topk_recon_losses):
        # Get min/max in log space for bins
        topk_recon_loss_nonzero = topk_recon_loss[topk_recon_loss > 0]
        bins = list(
            np.geomspace(
                topk_recon_loss_nonzero.min().item(),
                topk_recon_loss_nonzero.max().item(),
                100,
            )
        )
        ax.hist(
            topk_recon_loss.detach().cpu(),
            bins=bins,
            alpha=1 if i > 0 else 0.5,
            label=f"k={topks[i]}",
            histtype="step" if i > 0 else "bar",
        )
    ax.set_title(f"Topk recon loss for {data_generation_type} (batch_topk={batch_topk})")
    ax.set_xlabel("Recon loss")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_xscale("log")
    fig.show()
    plt.close(fig)


# %%
dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=False,  # Our labels will be the output of the target model
    data_generation_type=config.task_config.data_generation_type,
)
batch, labels = dataset.generate_batch(config.batch_size)
batch = batch.to(device)
labels = labels.to(device)

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

if torch.allclose(model.W_U.data, model.W_E.data.transpose(-2, -1)):
    print("W_E and W_U are tied")
else:
    print("W_E and W_U are not tied")


# %% Find subnet-feature-map


def top1_model_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk_mask: Float[Tensor, "batch n_instances k"] | None,
) -> SPDOutputs:
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


# %% Measure polysemanticity:

duplicity = {}  # subnet_idx -> number of features that use it
for subnet_idx in range(model.config.k):
    duplicity[subnet_idx] = len([f for f, s in subnet_indices.items() if s == subnet_idx])
duplicity_vals = np.array(list(duplicity.values()))
fig, ax = plt.subplots(figsize=(15, 5))
ax.hist(duplicity_vals, bins=[*np.arange(0, 10, 1)])
counts = np.bincount(duplicity_vals)
for i, count in enumerate(counts):
    if i == 0:
        name = "Dead"
    elif i == 1:
        name = "Monosemantic: "
    elif i == 2:
        name = "Duosemantic: "
    else:
        name = f"{i}-semantic: "
    ax.text(i + 0.5, count, name + str(count), ha="center", va="bottom")
fig.suptitle(f"Polysemanticity of model: {path}")
fig.show()

# %% Observe how well the model reconstructs the noise

instance_idx = 0
nrows = 1
feature_idx = 15
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(10, 1 + 3 * nrows))
fig.suptitle(f"Model {path}")
plot_resid_vs_mlp_out(
    target_model=target_model,
    device=device,
    ax=ax,
    instance_idx=instance_idx,
    feature_idx=feature_idx,
    topk_model_fn=top1_model_fn,
    subnet_indices=None,
)

# %% Linearity test: Enable one subnet after the other

n_features = model.config.n_features
feature_idx = 15
subtract_inputs = True  # TODO TRUE subnet
fig = plot_feature_response_with_subnets(
    topk_model_fn=top1_model_fn,
    device=device,
    model_config=model.config,
    feature_idx=feature_idx,
    subnet_idx=subnet_indices[feature_idx],
    batch_size=100,
)["feature_response_with_subnets"]
if fig is not None:
    fig.suptitle(f"Model {path}")
    plt.show()


# %% Feature-relu contribution plots

fig1, fig2 = plot_spd_relu_contribution(model, target_model, device, k_plot_limit=3)
fig1.suptitle("How much does each ReLU contribute to each feature?")
fig2.suptitle("How much does each feature route through each ReLU?")


# %% Individual feature response
def spd_model_fn(batch: Float[Tensor, "batch n_instances"]):
    assert config.topk is not None
    return run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        batch_topk=config.batch_topk,
        topk=config.topk,
        distil_from_target=config.distil_from_target,
    ).spd_topk_model_output


def target_model_fn(batch: Float[Tensor, "batch n_instances"]):
    return target_model(batch)[0]


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

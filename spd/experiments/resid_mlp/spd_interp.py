# %% Imports


from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    plot_resid_vs_mlp_out,
    plot_spd_relu_contribution,
    plot_virtual_weights_target_spd,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.experiments.resid_mlp.scaling_resid_mlp_training import naive_loss
from spd.run_spd import ResidualMLPTaskConfig, calc_recon_mse
from spd.utils import SPDOutputs, run_spd_forward_pass, set_seed

# %% Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed
# path = "wandb:spd-resid-mlp/runs/8st8rale"  # Dan's R1
# path = "wandb:spd-resid-mlp/runs/kkstog7o"  # Dan's R2
path = "wandb:spd-resid-mlp/runs/2ala9kjy"  # Stefan's initial try
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
    test_dataset = ResidualMLPDataset(
        n_instances=model.config.n_instances,
        n_features=model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        data_generation_type=data_generation_type,
    )
    instance_idx = 0
    batch, labels = test_dataset.generate_batch(batch_size)
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


# %% Measure polysemanticity:

# Dictionary feature_idx -> subnet_idx
subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

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
# TODO: Or bar chart for SPD/target, diff from goal (one-hot)

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
# TODO This could be little-violins. Or fill_betweens


# Dictionary feature_idx -> subnet_idx
subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

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
    plot_type="errorbar",
)["feature_response_with_subnets"]
if fig is not None:
    fig.suptitle(f"Model {path}")
    fig.savefig(f"feature_response_with_subnets_{feature_idx}.png", bbox_inches="tight", dpi=300)
    plt.show()


# %% Causal Scrubbing-esque test
def spd_model_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk: PositiveFloat = config.topk,
    batch_topk: bool = config.batch_topk,
) -> Float[Tensor, "batch n_instances n_features"]:
    assert config.topk is not None
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


# Dictionary feature_idx -> subnet_idx
subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

for data_generation_type in data_generation_types:
    batch_size = 5_000
    n_batches = 20
    test_dataset = ResidualMLPDataset(
        n_instances=model.config.n_instances,
        n_features=model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        data_generation_type=data_generation_type,
    )

    # Initialize tensors to store all losses
    all_loss_scrubbed = []
    all_loss_antiscrubbed = []
    all_loss_random = []
    all_loss_spd = []
    all_loss_zero = []

    for _ in tqdm(range(n_batches)):
        batch, labels = test_dataset.generate_batch(batch_size)
        batch = batch.to(device)
        active_features = torch.where(batch != 0)
        # Randomly assign 0 or 1 to topk mask
        random_topk_mask = torch.randint(
            0, 2, (batch_size, model.config.n_instances, model.config.k)
        )
        scrubbed_topk_mask = torch.randint(
            0, 2, (batch_size, model.config.n_instances, model.config.k)
        )
        antiscrubbed_topk_mask = torch.randint(
            0, 2, (batch_size, model.config.n_instances, model.config.k)
        )
        for b, i, f in zip(*active_features, strict=False):
            s = subnet_indices[f.item()]
            scrubbed_topk_mask[b, i, s] = 1
            antiscrubbed_topk_mask[b, i, s] = 0
        if data_generation_type == "at_least_zero_active":
            topk = config.topk
            batch_topk = config.batch_topk
        elif data_generation_type == "exactly_one_active":
            topk = 1
            batch_topk = True
        elif data_generation_type == "exactly_two_active":
            topk = 2
            batch_topk = True
        else:
            raise ValueError(f"Unknown data generation type: {data_generation_type}")
        out_spd = spd_model_fn(batch, topk=topk, batch_topk=batch_topk)
        out_random = top1_model_fn(batch, random_topk_mask).spd_topk_model_output
        out_scrubbed = top1_model_fn(batch, scrubbed_topk_mask).spd_topk_model_output
        out_antiscrubbed = top1_model_fn(batch, antiscrubbed_topk_mask).spd_topk_model_output
        out_target = target_model_fn(batch)
        # Calc MSE losses
        all_loss_scrubbed.append(
            ((out_scrubbed - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )
        all_loss_antiscrubbed.append(
            ((out_antiscrubbed - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )
        all_loss_random.append(
            ((out_random - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )
        all_loss_spd.append(((out_spd - out_target) ** 2).mean(dim=-1).flatten().detach().cpu())
        all_loss_zero.append(
            ((torch.zeros_like(out_target) - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )

    # Concatenate all batches
    loss_scrubbed = torch.cat(all_loss_scrubbed)
    loss_antiscrubbed = torch.cat(all_loss_antiscrubbed)
    loss_random = torch.cat(all_loss_random)
    loss_spd = torch.cat(all_loss_spd)
    loss_zero = torch.cat(all_loss_zero)

    # Print & plot the above
    loss_naive = naive_loss(
        n_features=model.config.n_features,
        d_mlp=model.config.d_mlp,
        p=config.task_config.feature_probability,
        bias=model.layers[0].linear1.bias is not None,
        embed="random",
    )

    print(f"Loss SPD:           {loss_spd.mean().item():.6f}")
    print(f"Loss scrubbed:      {loss_scrubbed.mean().item():.6f}")
    print(f"Loss antiscrubbed:  {loss_antiscrubbed.mean().item():.6f}")
    print(f"Loss naive:         {loss_naive:.6f}")
    print(f"Loss random:        {loss_random.mean().item():.6f}")
    print(f"Loss zero:          {loss_zero.mean().item():.6f}")

    # TODO orange bump: maybe SPD is using 2 subnets for some
    # features? Would explain scrubbed and antiscrubbed bimodality,
    # and random (which is just scrubbed + antiscrubbed) too.

    fig, ax = plt.subplots(figsize=(15, 5))
    log_bins = np.geomspace(1e-7, loss_zero.max().item(), 50).tolist()
    ax.hist(loss_spd, bins=log_bins, label="SPD", alpha=0.5)
    ax.hist(loss_scrubbed, bins=log_bins, label="Scrubbed", histtype="step")
    ax.hist(loss_antiscrubbed, bins=log_bins, label="Antiscrubbed", histtype="step")
    ax.hist(loss_random, bins=log_bins, label="Random", histtype="step")
    ax.hist(loss_zero, bins=log_bins, label="Zero", histtype="step")
    ax.axvline(loss_naive, color="black", linestyle="--", label="Naive")
    ax.legend()
    ax.set_ylabel(f"Count (out of {batch_size})")
    ax.set_xlabel("Recon loss")
    ax.set_xscale("log")
    fig.suptitle(f"Losses for {data_generation_type}")
    fig.show()


# %% "Forgetting"-style test for Lee. Let's say we want to ablate performance for all odd features,
# while preserving performance for even features.

batch_size = 1000
# target_model_train_config_dict
test_dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=True,  # Our labels will be the output of the target model
    label_type=target_model_train_config_dict["label_type"],
    act_fn_name=target_model_train_config_dict["resid_mlp_config"]["act_fn_name"],
    label_fn_seed=target_model_train_config_dict["label_fn_seed"],
    label_coeffs=target_label_coeffs,
    data_generation_type="at_least_zero_active",
)
batch, labels = test_dataset.generate_batch(batch_size)
batch = batch.to(device)
labels = labels.to(device)
instance_idx = 0

# Dictionary feature_idx -> subnet_idx
subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

subnets_corresponding_to_even_features = [
    subnet_indices[f] for f in range(model.config.n_features) if f % 2 == 0
]
topk_mask = torch.zeros_like(batch)
for subnet_idx in subnets_corresponding_to_even_features:
    topk_mask[:, instance_idx, subnet_idx] = 1

out_spd = spd_model_fn(batch)

out = top1_model_fn(batch, topk_mask=topk_mask)
out_target = target_model_fn(batch)
out_ablated = out.spd_topk_model_output
label_loss_target = (out_target - labels) ** 2
label_loss_spd = (out_spd - labels) ** 2
label_loss_ablated = (out_ablated - labels) ** 2
target_loss_spd = (out_target - out_spd) ** 2
target_loss_ablated = (out_target - out_ablated) ** 2
# Find samples in batch that contain only odd features
# odd_features = [i for i in range(model.config.n_features) if i % 2 == 1]
# even_features = [i for i in range(model.config.n_features) if i % 2 == 0]
odd_samples = torch.where(batch[:, instance_idx, 0::2].sum(dim=-1) != 0)[0].cpu().detach()
even_samples = torch.where(batch[:, instance_idx, 1::2].sum(dim=-1) != 0)[0].cpu().detach()
# Exclude samples containing both odd and even features

both_odd_and_even_samples = np.intersect1d(odd_samples.numpy(), even_samples.numpy())
only_odd_samples = np.setdiff1d(odd_samples.numpy(), both_odd_and_even_samples)
only_even_samples = np.setdiff1d(even_samples.numpy(), both_odd_and_even_samples)

label_loss_target_odd = (
    label_loss_target[only_odd_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
label_loss_spd_odd = label_loss_spd[only_odd_samples].mean(dim=-1).flatten().detach().cpu().numpy()  # noqa: E501
label_loss_ablated_odd = (
    label_loss_ablated[only_odd_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
label_loss_target_even = (
    label_loss_target[only_even_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
label_loss_spd_even = (
    label_loss_spd[only_even_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
label_loss_ablated_even = (
    label_loss_ablated[only_even_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
target_loss_spd_odd = (
    target_loss_spd[only_odd_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
target_loss_spd_even = (
    target_loss_spd[only_even_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
target_loss_ablated_odd = (
    target_loss_ablated[only_odd_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
target_loss_ablated_even = (
    target_loss_ablated[only_even_samples].mean(dim=-1).flatten().detach().cpu().numpy()
)  # noqa: E501
target_loss_target_odd = torch.zeros_like(torch.tensor(target_loss_ablated_odd)).numpy()
target_loss_target_even = torch.zeros_like(torch.tensor(target_loss_ablated_even)).numpy()
# Boxplot chart
# Create a dataframe for seaborn
data = pd.DataFrame(
    {
        "Label Loss": np.concatenate(
            [
                label_loss_target_odd,
                label_loss_spd_odd,
                label_loss_ablated_odd,
                label_loss_target_even,
                label_loss_spd_even,
                label_loss_ablated_even,
            ]
        ),
        "Target Loss": np.concatenate(
            [
                target_loss_target_odd,
                target_loss_spd_odd,
                target_loss_ablated_odd,
                target_loss_target_even,
                target_loss_spd_even,
                target_loss_ablated_even,
            ]
        ),
        "Model": np.repeat(
            ["Target", "APD", "Ablated", "Target", "APD", "Ablated"],
            [
                len(label_loss_target_odd),
                len(label_loss_spd_odd),
                len(label_loss_ablated_odd),
                len(label_loss_target_even),
                len(label_loss_spd_even),
                len(label_loss_ablated_even),
            ],
        ),
        "Sample Type": np.repeat(
            ["Odd", "Odd", "Odd", "Even", "Even", "Even"],
            [
                len(label_loss_target_odd),
                len(label_loss_ablated_odd),
                len(label_loss_spd_odd),
                len(label_loss_target_even),
                len(label_loss_ablated_even),
                len(label_loss_spd_even),
            ],
        ),
    }
)

fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
axes = np.atleast_1d(axes)  # type: ignore
ax = axes[0]
sns.boxplot(data=data, x="Sample Type", y="Label Loss", hue="Model", ax=ax)
ax.set_yscale("log")
ax.set_ylim(bottom=1e-5)
ax.set_title("Label loss")
ax.set_ylabel("")
# ax.axhline(y=loss_naive, color="k", linestyle="--", label="Monosemantic neuron solution", alpha=0.5)
ax.legend(bbox_to_anchor=(0.5, -0.05), loc="upper center", bbox_transform=fig.transFigure, ncol=3)

ax = axes[1]
sns.boxplot(data=data, x="Sample Type", y="Target Loss", hue="Model", ax=ax)
ax.set_yscale("log")
ax.set_ylim(bottom=1e-6)
ax.set_title("Target loss")
ax.set_ylabel("")
ax.legend().remove()
fig.savefig("ablation_story.png", bbox_inches="tight", dpi=300)
fig.show()


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

# %% Feature-relu contribution plots

fig1, fig2 = plot_spd_relu_contribution(model, target_model, device, k_plot_limit=3)
fig1.suptitle("How much does each ReLU contribute to each feature?")
fig2.suptitle("How much does each feature route through each ReLU?")


# %% Virtual weights
fig = plot_virtual_weights_target_spd(target_model, model, device)
fig.show()

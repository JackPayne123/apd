# %%


from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from torch import Tensor

from spd.experiments.resid_linear.models import ResidualLinearModel
from spd.experiments.resid_linear.resid_linear_dataset import ResidualLinearDataset
from spd.run_spd import ResidualLinearConfig, calc_recon_mse
from spd.utils import run_spd_forward_pass, set_seed

# %%

if __name__ == "__main__":
    # Set up device and seed
    device = "cpu"
    print(f"Using device: {device}")
    set_seed(0)  # You can change this seed if needed

    # Load model and config
    model, task_config, label_coeffs = ResidualLinearModel.from_pretrained(
        # "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/resid_linear/out/resid_linear_identity_n-features400_d-resid200_d-mlp300_n-layers1_seed0/target_model.pth"
        "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/resid_linear/out/resid_linear_identity_n-features100_d-resid200_d-mlp50_n-layers1_seed0/target_model.pth"
    )
    print(task_config)
    model = model.to(device)
    task_config["batch_size"] = 128

    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=model.n_features,
        feature_probability=task_config["feature_probability"],
        device=device,
        label_coeffs=label_coeffs,
        data_generation_type="exactly_one_active",
    )
    batch, labels = dataset.generate_batch(task_config["batch_size"])
    # batch2, labels2 = dataset.generate_batch(task_config["batch_size"])
    # batch = batch + batch2
    # labels = labels + labels2
    # Print some basic information about the model
    # print(f"Model structure:\n{model}")
    print(f"Number of features: {model.n_features}")
    print(f"Embedding dimension: {model.d_embed}")
    print(f"MLP dimension: {model.d_mlp}")
    print(f"Number of layers: {model.n_layers}")

    plt.figure(figsize=(20, 5))
    plt.suptitle("First 3 features")
    plt.subplot(211)
    plt.title("Input layer, W_E @ W_in.T")
    plt.imshow(
        (model.W_E @ model.layers[0].input_layer.weight.T)[:5].cpu().detach(),
        cmap="RdBu",
        norm=CenteredNorm(),
    )
    plt.colorbar()
    plt.subplot(212)
    plt.title("Output layer, W_E @ W_out")
    plt.imshow(
        (model.W_E @ model.layers[0].output_layer.weight)[:5].cpu().detach(),
        cmap="RdBu",
        norm=CenteredNorm(),
    )
    plt.colorbar()
    plt.show()

    # batch: [batch_size, n_features]
    # Set 1st feature to 1
    batch[:, 2] = 0.5
    # batch[:, 1:] = 0
    out, pre_acts, post_acts = model(batch)
    print(f"MSE: {F.mse_loss(out, labels).item()}")
    embed = pre_acts["layers.0.input_layer.weight"]
    pre_relu = post_acts["layers.0.input_layer.weight"]
    post_relu = pre_acts["layers.0.output_layer.weight"]
    print(f"Pre-relu shape: {pre_relu.shape}")
    print(f"Post-relu shape: {post_relu.shape}")
    # Imshow pre-relu and post-relu
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    # axs[0].imshow(
    #     pre_relu.cpu().detach().mean(dim=0, keepdim=True), cmap="RdBu", norm=CenteredNorm()
    # )
    random_direction = torch.randn(model.d_mlp).to(device) * 10
    axs[0].imshow(
        pre_relu.T.cpu().detach()[:, -200:],  # + 0.01 * random_direction.unsqueeze(0).T,
        cmap="RdBu",
        norm=CenteredNorm(),
    )
    axs[1].imshow(post_relu.T.cpu().detach()[:, -200:], cmap="RdBu", norm=CenteredNorm())
    plt.show()

    # Calculate MMCS (mean max cosine similarity) of outputs with labels
    dot_products = einops.einsum(out - embed, model.W_E, "b d, i d -> b i")
    # Scatter dot_products vs batch, both flattened
    # plt.title("MLP_out vs Batch")
    # plt.scatter(batch[:].flatten().cpu().detach(), dot_products[:].flatten().cpu().detach())
    # plt.xlabel("batch val")
    # plt.ylabel("mlp_out W_E dot prod")
    # plt.show()

    plt.figure(figsize=(30, 30))
    for f in range(model.n_features):
        plt.subplot(10, 10, f + 1)
        plt.title(f"MLP_out vs Batch, feature {f}")
        cs = torch.zeros_like(batch[:, :])
        cs[torch.where(batch[:, f] > 0)[0], :] = 1
        print(f"How often was feature {f} > 0: {(batch[:, f] > 0).sum().item()}")
        plt.scatter(
            batch[:].flatten().cpu().detach(),
            dot_products[:].flatten().cpu().detach(),
            c=cs,
            marker=".",
            s=2,
            alpha=0.2,
        )
        plt.xlabel("batch val")
        plt.ylabel("mlp_out W_E dot prod")


# %%
plt.hist(post_relu.mean(dim=0).detach().cpu().numpy())

# %% Weight analysis: For each feature, how many ReLUs does it affect?


W_E: Float[Tensor, "n_features d_embed"] = model.W_E.cpu().detach()
W_in: Float[Tensor, "d_mlp d_embed"] = model.layers[0].input_layer.weight.cpu().detach()
W_out: Float[Tensor, "d_embed d_mlp"] = model.layers[0].output_layer.weight.cpu().detach()
assert W_E.shape == (model.n_features, model.d_embed)
assert W_in.shape == (model.d_mlp, model.d_embed)
assert W_out.shape == (model.d_embed, model.d_mlp)

in_conns: Float[Tensor, "n_features d_mlp"] = einops.einsum(
    W_E, W_in, "n_features d_embed, d_mlp d_embed -> n_features d_mlp"
)
out_conns: Float[Tensor, "d_mlp n_features"] = einops.einsum(
    W_out, W_E, "d_embed d_mlp, n_features d_embed -> d_mlp n_features"
)
conns: Float[Tensor, "n_features n_features"] = einops.einsum(
    in_conns, out_conns, "n_features_in d_mlp, d_mlp n_features_out -> n_features_in n_features_out"
)

plt.hist(conns.flatten().detach().cpu().numpy(), bins=30)
plt.semilogy()
plt.show()
# How many conns > 0.5
print(f"{(conns > 0.5).sum().item()} connections > 0.5")

plt.matshow(conns.cpu().detach().T)
plt.title("Connection strength between features")
plt.xlabel("Feature input")
plt.ylabel("Feature readoff")
plt.colorbar()
plt.show()

# %% Conns via relus
relu_conns: Float[Tensor, "n_features_in d_mlp n_features_out"] = einops.einsum(
    in_conns,
    out_conns,
    "n_features_in d_mlp, d_mlp n_features_out -> n_features_in n_features_out d_mlp",
)
for f in range(5):
    plt.plot(relu_conns[f, f].detach().cpu().numpy(), label=f"Feature {f}")
plt.legend()
plt.xlabel("ReLUs")
plt.ylabel("Connection strength")
plt.show()

# %%
relu_conns_diag: Float[Tensor, "n_features d_mlp"] = einops.einsum(
    in_conns,
    out_conns,
    "n_features d_mlp, d_mlp n_features-> n_features d_mlp",
)
biggest_relu_conns = relu_conns_diag.abs().max(dim=1).values
# Histogram of biggest relu conn per feature
plt.hist(
    biggest_relu_conns.flatten().detach().cpu().numpy(),
    bins=30,
    range=(0, 1),
    histtype="step",
    label="biggest relu conn per feature",
)
# Histogram of sum of relu conns
plt.hist(
    relu_conns_diag.sum(dim=1).flatten().detach().cpu().numpy(),
    bins=30,
    range=(0, 1),
    histtype="step",
    label="sum of relu conns",
)
plt.legend()
plt.show()

# %%
# For each feature, how many topk ReLUs do I need to explain 90% of the connection?
topk_90_percent = []
for f in range(model.n_features):
    c = relu_conns_diag[f].sort(descending=True).values
    cumsum = c.cumsum(dim=0)
    threshold = 0.9 * c.sum().item()
    topk = (cumsum < threshold).sum().item()
    topk_90_percent.append(topk)
plt.hist(
    topk_90_percent, bins=30, label="Topk ReLUs needed to explain 90% of the connection", alpha=0.5
)

topk_80_percent = []
for f in range(model.n_features):
    c = relu_conns_diag[f].sort(descending=True).values
    cumsum = c.cumsum(dim=0)
    threshold = 0.8 * c.sum().item()
    topk = (cumsum < threshold).sum().item()
    topk_80_percent.append(topk)
plt.hist(
    topk_80_percent, bins=30, label="Topk ReLUs needed to explain 80% of the connection", alpha=0.5
)
plt.legend()
plt.xlabel("Topk ReLUs")
plt.ylabel("Count")
plt.show()

# %%

# I'd like to know whether certain ReLUs are specialized to be mono-feature ReLUs while others
# specialize in being used for multiple features.
# Make a violin plot for each ReLU

# For every ReLU (x val) scatter (y val) the connection strengths
plt.figure(figsize=(20, 5))
plt.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
for i in range(model.d_mlp):
    plt.scatter([i] * 100, relu_conns_diag[:, i], alpha=0.3, marker=".")
    plt.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
plt.axhline(0, color="k", linestyle="--", alpha=0.3)
plt.xlabel("ReLUs")
plt.ylabel("Connection strength for all features")
plt.show()

# Same plot but for every scatter dot > 0.2 add a plt.text in same color with the index of the feature
plt.figure(figsize=(20, 5))
plt.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
for i in range(model.d_mlp):
    cmap_scatter = plt.get_cmap("tab20")
    color = cmap_scatter(i / model.d_mlp)
    plt.scatter([i] * 100, relu_conns_diag[:, i], alpha=0.3, marker=".", c="k")
    plt.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for j in range(model.n_features):
        if relu_conns_diag[j, i] > 0.2:
            cmap_label = plt.get_cmap("hsv")
            plt.text(i, relu_conns_diag[j, i], str(j), color=cmap_label(j / model.n_features))
plt.axhline(0, color="k", linestyle="--", alpha=0.3)
plt.xlabel("ReLUs")
plt.ylabel("Connection strength for all features")
plt.show()

# %%
# Inverted version of the plot above: For each feature, plot the connection strength of the topk
# ReLUs
plt.figure(figsize=(40, 5))
plt.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
for i in range(model.n_features):
    plt.scatter([i] * 50, relu_conns_diag[i, :], alpha=0.3, marker=".", c="k")
    plt.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for j in range(model.d_mlp):
        if relu_conns_diag[i, j] > 0.2:
            cmap_label = plt.get_cmap("hsv")
            plt.text(i, relu_conns_diag[i, j], str(j), color=cmap_label(j / model.d_mlp))
plt.axhline(0, color="k", linestyle="--", alpha=0.3)
plt.xlabel("Features")
plt.ylabel("Connection strength for all ReLUs")
plt.show()


# %%
# For each relu, how many topk features does it provide >80% of the connection?
threshold = 0.1
topk_80_percent_feats_per_relu = []
for r in range(model.d_mlp):
    # How many features does this ReLU provide > 90% to?
    this_relu_feats = 0
    for f in range(model.n_features):
        total_conn = relu_conns_diag[f, :].sum().item()
        this_relu_conn = relu_conns_diag[f, r]
        if this_relu_conn / total_conn > threshold:
            this_relu_feats += 1
    topk_80_percent_feats_per_relu.append(this_relu_feats)
    if this_relu_feats > 0:
        print(f"ReLU {r} provides >{threshold:.0%} to {this_relu_feats} features")
plt.hist(
    topk_80_percent_feats_per_relu,
    bins=30,
    # label="Topk features per ReLU that provide >80% of the connection",
    alpha=0.5,
)
plt.ylabel("Count (over ReLUs)")
plt.xlabel(f"How many features does this ReLU provide >{threshold:.0%} of the connection to?")
# %%
# Hist relu_conns_diag

plt.hist(relu_conns_diag.flatten().detach().cpu().numpy(), bins=30)
plt.semilogy()
plt.show()


# %%

# in_conns: Float[Tensor, "n_features d_mlp"] = einops.einsum(
#      "n_features d_embed, d_mlp d_embed -> n_features d_mlp"
# )
# out_conns: Float[Tensor, "d_mlp n_features"] = einops.einsum(
#     W_out, W_E, "d_embed d_mlp, n_features d_embed -> d_mlp n_features"
# )
# relu_conns_diag2: Float[Tensor, "n_features d_mlp"] = einops.einsum(
#     W_E, W_in, W_out, W_E,
#     "n_features d_embed1, d_mlp d_embed1, d_embed2 d_mlp, n_features d_embed2-> n_features d_mlp",
# )
W_E_fake = torch.randn(model.n_features, model.d_embed)
W_E_fake = W_E_fake / W_E_fake.norm(dim=-1, keepdim=True)
relu_conns_diag_fake: Float[Tensor, "n_features d_mlp"] = einops.einsum(
    W_E_fake,
    W_in,
    W_out,
    W_E_fake,
    "n_features d_embed1, d_mlp d_embed1, d_embed2 d_mlp, n_features d_embed2-> n_features d_mlp",
)
# Make a baseline plot, how would random feature directions look like?
plt.figure(figsize=(40, 5))
plt.title("RANDOM BASELINE")
plt.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
for i in range(model.n_features):
    plt.scatter([i] * 50, relu_conns_diag_fake[i, :], alpha=0.3, marker=".", c="k")
    plt.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for j in range(model.d_mlp):
        if relu_conns_diag_fake[i, j] > 0.2:
            cmap_label = plt.get_cmap("hsv")
            plt.text(i, relu_conns_diag_fake[i, j], str(j), color=cmap_label(j / model.d_mlp))
plt.axhline(0, color="k", linestyle="--", alpha=0.3)
plt.xlabel("Features")
plt.ylabel("Connection strength for all ReLUs")
plt.show()
# %%

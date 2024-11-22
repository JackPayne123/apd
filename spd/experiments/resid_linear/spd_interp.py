import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from torch import Tensor
from tqdm import tqdm

from spd.experiments.resid_linear.models import ResidualLinearModel
from spd.experiments.resid_linear.resid_linear_dataset import ResidualLinearDataset
from spd.utils import set_seed

# %%

# Set up device and seed
device = "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed

# Load model and config
path = (
    "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/resid_linear/out/"
    "resid_linear_identity_n-features100_d-resid100_d-mlp50_n-layers1_seed0/target_model.pth"
)
model, task_config, label_coeffs = ResidualLinearModel.from_pretrained(path)

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


print(f"Number of features: {model.n_features}")
print(f"Embedding dimension: {model.d_embed}")
print(f"MLP dimension: {model.d_mlp}")
print(f"Number of layers: {model.n_layers}")

k = model.n_features
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

relu_conns: Float[Tensor, "n_features_in d_mlp n_features_out"] = einops.einsum(
    in_conns,
    out_conns,
    "n_features_in d_mlp, d_mlp n_features_out -> n_features_in n_features_out d_mlp",
)
relu_conns_diag: Float[Tensor, "n_features d_mlp"] = einops.einsum(
    in_conns,
    out_conns,
    "n_features d_mlp, d_mlp n_features-> n_features d_mlp",
)
relu_conns_diag_filtered: Float[Tensor, "n_features d_mlp"] = einops.einsum(
    in_conns * (in_conns > 0).float(),
    out_conns * (out_conns > 0).float(),
    "n_features d_mlp, d_mlp n_features-> n_features d_mlp",
)

# %%
# Inverted version of the plot above: For each feature, plot the connection strength of the topk
# ReLUs
# Stefan's fav plot
plt.figure(figsize=(40, 5))
plt.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
for i in range(model.n_features):
    plt.scatter([i] * model.d_mlp, relu_conns_diag[i, :], alpha=0.3, marker=".", c="k")
    plt.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for j in range(model.d_mlp):
        if relu_conns_diag[i, j] > 0.12:
            cmap_label = plt.get_cmap("hsv")
            plt.text(i, relu_conns_diag[i, j], str(j), color=cmap_label(j / model.d_mlp))
plt.axhline(0, color="k", linestyle="--", alpha=0.3)
plt.xlabel("Features")
plt.ylabel("Connection strength for all ReLUs")
plt.show()

# %%
# W_in version:
# ReLUs
# Stefan's fav plot
plt.figure(figsize=(40, 5))
plt.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
for i in range(model.n_features):
    plt.scatter([i] * model.d_mlp, in_conns[i, :], alpha=0.3, marker=".", c="k")
    plt.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for j in range(model.d_mlp):
        if in_conns[i, j] > 0.12:
            cmap_label = plt.get_cmap("hsv")
            plt.text(i, in_conns[i, j], str(j), color=cmap_label(j / model.d_mlp))
plt.axhline(0, color="k", linestyle="--", alpha=0.3)
plt.xlabel("Features")
plt.ylabel("IN-Connection strength for all ReLUs")
plt.show()

# %%
# sub_W_in: Float[Tensor, "k d_mlp d_embed"] = torch.zeros((k, model.d_mlp, model.d_embed))
# sub_W_out: Float[Tensor, "k d_embed d_mlp"] = torch.zeros((k, model.d_embed, model.d_mlp))
# for i in range(k):
#     # Select the part of W_in that is responsible for the i-th feature
#     for j in range(model.d_mlp):
#         # Desired connection between the i-th feature and the j-th ReLU
#         desired_conn = relu_conns_diag[i, j]

# %%

spd_path = (
    "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/resid_linear/out/"
    "rp_p8.00e-01_topk1.15e+00_topkrecon1.00e+00_schatten1.00e-01_sd0_attr-gra_lr1.00e-02_bs1024_ft100_lay1_resid100_mlp50/model_30000.pth"
)

spd_weights = torch.load(spd_path, weights_only=True)

W_in_A: Float[Tensor, "k d_embed m1"] = spd_weights["layers.0.linear1.A"].cpu().detach()
W_in_B: Float[Tensor, "k m1 d_mlp"] = spd_weights["layers.0.linear1.B"].cpu().detach()
W_out_A: Float[Tensor, "k d_mlp m2"] = spd_weights["layers.0.linear2.A"].cpu().detach()
W_out_B: Float[Tensor, "k m2 d_embed"] = spd_weights["layers.0.linear2.B"].cpu().detach()

W_in_k: Float[Tensor, "k d_embed d_mlp"] = einops.einsum(
    W_in_A, W_in_B, "k d_embed m1, k m1 d_mlp -> k d_embed d_mlp"
)
W_out_k: Float[Tensor, "k d_mlp d_embed"] = einops.einsum(
    W_out_A, W_out_B, "k d_mlp m2, k m2 d_embed -> k d_mlp d_embed"
)


in_conns_k: Float[Tensor, "k n_features d_mlp"] = einops.einsum(
    W_E, W_in_k, "n_features d_embed, k d_embed d_mlp -> k n_features d_mlp"
)
out_conns_k: Float[Tensor, "k d_mlp n_features"] = einops.einsum(
    W_out_k, W_E, "k d_mlp d_embed, n_features d_embed -> k d_mlp n_features"
)

conns_k: Float[Tensor, "k n_features n_features"] = einops.einsum(
    in_conns_k,
    out_conns_k,
    "k n_features_in d_mlp, k d_mlp n_features_out -> k n_features_in n_features_out",
)

# relu_conns_k: Float[Tensor, "k n_features_in d_mlp n_features_out"] = einops.einsum(
#     in_conns_k,
#     out_conns_k,
#     "k n_features_in d_mlp, k d_mlp n_features_out -> k n_features_in n_features_out d_mlp",
# )
relu_conns_diag_k: Float[Tensor, "k n_features d_mlp"] = einops.einsum(
    in_conns_k,
    out_conns_k,
    "k n_features d_mlp, k d_mlp n_features-> k n_features d_mlp",
)
relu_conns_diag_k_filtered: Float[Tensor, "k n_features d_mlp"] = einops.einsum(
    in_conns_k * (in_conns_k > 0).float(),
    out_conns_k * (out_conns_k > 0).float(),
    "k n_features d_mlp, k d_mlp n_features-> k n_features d_mlp",
)

relu_conns_diag_k_sum = relu_conns_diag_k.sum(dim=0)
in_conns_k_sum = in_conns_k.sum(dim=0)
out_conns_k_sum = out_conns_k.sum(dim=0)
relu_conns_diag_k_sum2 = einops.einsum(
    in_conns_k_sum,
    out_conns_k_sum,
    "n_features d_mlp, d_mlp n_features -> n_features d_mlp",
)
# torch.testing.assert_close(relu_conns_diag_k_sum, relu_conns_diag_k_sum2)

# %%

for ki in tqdm(range(-2, k)):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_xlabel("Features")
    ax.set_ylabel(f"k = {ki}")
    ax.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(model.n_features):
        ax.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        if ki == -2:
            ax.set_ylabel("Full W_in and W_out product")
            ax.scatter([i] * model.d_mlp, relu_conns_diag[i, :], alpha=0.3, marker=".", c="k")
            for j in range(model.d_mlp):
                if relu_conns_diag[i, j] > 0.12:
                    cmap_label = plt.get_cmap("hsv")
                    ax.text(i, relu_conns_diag[i, j], str(j), color=cmap_label(j / model.d_mlp))

        if ki == -1:
            ax.set_ylabel("Sum over k without cross-terms")
            relu_conns_diag_sum = relu_conns_diag_k.sum(dim=0)
            ax.scatter([i] * model.d_mlp, relu_conns_diag_sum[i, :], alpha=0.3, marker=".", c="k")
            for j in range(model.d_mlp):
                if relu_conns_diag_sum[i, j] > 0.12:
                    cmap_label = plt.get_cmap("hsv")
                    ax.text(i, relu_conns_diag_sum[i, j], str(j), color=cmap_label(j / model.d_mlp))
        else:
            ax.scatter([i] * model.d_mlp, relu_conns_diag_k[ki, i, :], alpha=0.3, marker=".", c="k")
            for j in range(model.d_mlp):
                if relu_conns_diag_k[ki, i, j] > 0.12:
                    cmap_label = plt.get_cmap("hsv")
                    ax.text(
                        i, relu_conns_diag_k[ki, i, j], str(j), color=cmap_label(j / model.d_mlp)
                    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    fig.savefig(f"relu_conns_diag_k_{ki}.png")
    plt.close(fig)

# %% Filtered

for ki in range(-2, 0):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_xlabel("Features")
    ax.set_ylabel(f"k = {ki}")
    ax.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(model.n_features):
        ax.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        if ki == -2:
            ax.set_ylabel("Full W_in and W_out product")
            ax.scatter(
                [i] * model.d_mlp, relu_conns_diag_filtered[i, :], alpha=0.3, marker=".", c="k"
            )
            for j in range(model.d_mlp):
                if relu_conns_diag_filtered[i, j] > 0.12:
                    cmap_label = plt.get_cmap("hsv")
                    ax.text(
                        i, relu_conns_diag_filtered[i, j], str(j), color=cmap_label(j / model.d_mlp)
                    )

        if ki == -1:
            ax.set_ylabel("Sum over k without cross-terms")
            relu_conns_diag_sum_filtered = relu_conns_diag_k.sum(dim=0)
            ax.scatter(
                [i] * model.d_mlp, relu_conns_diag_sum_filtered[i, :], alpha=0.3, marker=".", c="k"
            )
            for j in range(model.d_mlp):
                if relu_conns_diag_sum_filtered[i, j] > 0.12:
                    cmap_label = plt.get_cmap("hsv")
                    ax.text(
                        i,
                        relu_conns_diag_sum_filtered[i, j],
                        str(j),
                        color=cmap_label(j / model.d_mlp),
                    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    print(f"Saved {ki}")
    fig.savefig(f"filtered_relu_conns_diag_k_{ki}.png")
    plt.close("all")

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
fig.suptitle("Relu conns")
axes[0].matshow(relu_conns_diag_k.sum(dim=0))
axes[1].matshow(relu_conns_diag)
plt.show()
# in_conns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
fig.suptitle("in_conns")
axes[0].matshow(in_conns_k.sum(dim=0))
axes[1].matshow(in_conns)
plt.show()
# out_conns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
fig.suptitle("out_conns")
axes[0].matshow(out_conns_k.sum(dim=0))
axes[1].matshow(out_conns)
plt.show()

# Win
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
fig.suptitle("W_in")
axes[0].matshow(W_in_k.sum(dim=0))
axes[1].matshow(W_in.T)
plt.show()
# Wout
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
fig.suptitle("W_out")
axes[0].matshow(W_out_k.sum(dim=0))
axes[1].matshow(W_out.T)
plt.show()


# %% Idea: Ablation test
W_E: Float[Tensor, "n_features d_embed"] = model.W_E.cpu().detach()
W_in: Float[Tensor, "d_mlp d_embed"] = model.layers[0].input_layer.weight.cpu().detach()
W_out: Float[Tensor, "d_embed d_mlp"] = model.layers[0].output_layer.weight.cpu().detach()
batch, labels = dataset.generate_batch(task_config["batch_size"])
# For each feature
for f in range(3):
    batch_f = torch.zeros_like(batch)
    batch_f[:, f] = -1 + 2 * torch.rand_like(batch[:, f])
    out_f, pre_acts, _ = model(batch_f)
    embeds = pre_acts["layers.0.input_layer.weight"]
    mlp_out_f = out_f - embeds
    feature_out_f = einops.einsum(
        mlp_out_f, model.W_E, "batch d_embed, n_features d_embed -> batch n_features"
    )
    # plt.scatter(
    #     batch_f[:, f].detach().cpu().numpy(), feature_out_f[:, f].detach().cpu().numpy(), color="k"
    # )
    mse = ((feature_out_f[:, f] - batch_f[:, f]) ** 2).mean()
    # Now do a run without using the model:
    resid_pre_mlp = einops.einsum(
        batch_f, W_E, "batch n_features, n_features d_embed -> batch d_embed"
    )
    pre_relu = einops.einsum(resid_pre_mlp, W_in, "batch d_embed, d_mlp d_embed -> batch d_mlp")
    post_relu = torch.relu(pre_relu)
    mlp_out = einops.einsum(post_relu, W_out, "batch d_mlp, d_embed d_mlp -> batch d_embed")
    feature_out = einops.einsum(
        mlp_out, W_E, "batch d_embed, n_features d_embed -> batch n_features"
    )
    # plt.scatter(
    #     batch_f[:, f].detach().cpu().numpy(),
    #     feature_out[:, f].detach().cpu().numpy(),
    #     color="r",
    #     s=1,
    # )
    # plt.show()
    mse_orig = ((feature_out[:, f] - batch_f[:, f]) ** 2).mean()
    # print(f"MSE for feature {f}: {mse}")
    # Ablate a neuron
    mse_diffs = torch.zeros(model.d_mlp)
    for n in range(model.d_mlp):
        post_relu_n = post_relu.clone()
        post_relu_n[:, n] = 0
        mlp_out_n = einops.einsum(post_relu_n, W_out, "batch d_mlp, d_embed d_mlp -> batch d_embed")
        feature_out_n = einops.einsum(
            mlp_out_n, W_E, "batch d_embed, n_features d_embed -> batch n_features"
        )
        mse_n = ((feature_out_n[:, f] - batch_f[:, f]) ** 2).mean()
        mse_diffs[n] = mse_n - mse_orig
        # print(f"MSE diff for feature {f} ablated neuron {n}: {mse_n-mse_orig: .5f}")
    plt.scatter([f] * model.d_mlp, mse_diffs.detach().cpu().numpy(), color="k", marker=".", s=3)
    # Add text again
    for n in range(model.d_mlp):
        if mse_diffs[n] > 0.002:
            cmap_label = plt.get_cmap("hsv")
            plt.text(f, mse_diffs[n], str(n), color=cmap_label(n / model.d_mlp))
plt.show()
# %%

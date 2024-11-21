# %%


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
    "resid_linear_identity_n-features100_d-resid200_d-mlp50_n-layers1_seed0/target_model.pth"
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
# %%
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
plt.suptitle("First 5 features")
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
# batch[:, 2] = 0.5
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
plt.title("MLP_out vs Batch")
plt.scatter(batch[:].flatten().cpu().detach(), dot_products[:].flatten().cpu().detach())
plt.xlabel("batch val")
plt.ylabel("mlp_out W_E dot prod")
plt.show()

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

# %%

if model.d_mlp == 2:
    # in_conns: Float[Tensor, "n_features d_mlp"] = einops.einsum(
    plt.figure(figsize=(5, 5))
    plt.title("in_conns")
    for i in range(model.n_features):
        c = f"C{i}"
        plt.arrow(
            0,
            0,
            in_conns[i, 0].item(),
            in_conns[i, 1].item(),
            head_width=0.01,
            head_length=0.01,
            color=c,
        )
    plt.show()
    # out_conns: Float[Tensor, "d_mlp n_features"] = einops.einsum(
    plt.figure(figsize=(5, 5))
    plt.title("out_conns")
    for i in range(model.n_features):
        c = f"C{i}"
        plt.arrow(
            0,
            0,
            out_conns.T[i, 0].item(),
            out_conns.T[i, 1].item(),
            head_width=0.1,
            head_length=0.1,
            color=c,
        )
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
    cmap_scatter = plt.get_cmap("tab20")
    color = cmap_scatter(i / model.d_mlp)
    plt.scatter([i] * model.n_features, relu_conns_diag[:, i], alpha=0.3, marker=".", c="k")
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
    plt.scatter([i] * model.d_mlp, relu_conns_diag[i, :], alpha=0.3, marker=".", c="k")
    plt.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for j in range(model.d_mlp):
        if relu_conns_diag[i, j] > 0.1:
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
    plt.scatter([i] * model.d_mlp, relu_conns_diag_fake[i, :], alpha=0.3, marker=".", c="k")
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

# Evaluate the model a bit. First MSE:
dataset = ResidualLinearDataset(
    embed_matrix=model.W_E,
    n_features=model.n_features,
    feature_probability=task_config["feature_probability"],
    device=device,
    label_coeffs=label_coeffs,
    data_generation_type="exactly_one_active",
)

batch, labels = dataset.generate_batch(task_config["batch_size"])

model, task_config, label_coeffs = ResidualLinearModel.from_pretrained(path)
out, pre_acts, post_acts = model(batch)
print(f"MSE (vanilla): {F.mse_loss(out, labels).item()}")

if model.n_features == model.d_embed:
    model.W_E.data = torch.zeros_like(model.W_E)
    for i in range(model.n_features):
        model.W_E.data[i, i] = 1
    model.layers[0].input_layer.weight.data = torch.zeros_like(
        model.layers[0].input_layer.weight.data
    )

    model.layers[0].input_layer.weight.data[0, 0] = 1.0
    model.layers[0].input_layer.weight.data[1, 1] = 1.0
    if model.layers[0].input_layer.bias is not None:
        model.layers[0].input_layer.bias.data = torch.zeros_like(
            model.layers[0].input_layer.bias.data
        )

    model.layers[0].output_layer.weight.data = torch.zeros_like(
        model.layers[0].output_layer.weight.data
    )
    model.layers[0].output_layer.weight.data[0, 0] = 1.0
    model.layers[0].output_layer.weight.data[1, 1] = 1.0
    if model.layers[0].output_layer.bias is not None:
        model.layers[0].output_layer.bias.data = torch.zeros_like(
            model.layers[0].output_layer.bias.data
        )

    out, pre_acts, post_acts = model(batch)
    print(f"MSE (handcoded): {F.mse_loss(out, labels).item()}")
    model, task_config, label_coeffs = ResidualLinearModel.from_pretrained(path)

embed = pre_acts["layers.0.input_layer.weight"]
dot_products = einops.einsum(out - embed, model.W_E, "b d, i d -> b i")
plt.title("MLP_out vs Batch")
plt.scatter(batch[:].flatten().cpu().detach(), dot_products[:].flatten().cpu().detach())
plt.xlabel("batch val")
plt.ylabel("mlp_out W_E dot prod")
plt.show()
# %%
# Now check how well MSE is for each individual feature
for f in range(model.n_features):
    batch_f = torch.zeros_like(batch)
    batch_f[:, f] = -1 + 2 * torch.rand_like(batch[:, f])
    out_f, pre_acts_f, _ = model(batch_f)
    embed_f = pre_acts_f["layers.0.input_layer.weight"]
    labels_f = embed_f + torch.relu(model.W_E[f] * 1.0)
    mse_f = F.mse_loss(out_f, labels_f).item()
    print(f"MSE for feature {f}: {mse_f}")
    dot_products_f = einops.einsum(out_f - embed_f, model.W_E, "b d, i d -> b i")
    plt.scatter(batch_f[:].flatten().cpu().detach(), dot_products_f[:].flatten().cpu().detach())
plt.show()
# %%
# # Check how well it does if 2 features are active
# batch, labels = dataset.generate_batch(task_config["batch_size"])
# for f in range(model.n_features):
#     batch_f = batch.clone()
#     batch_f[:, f] = -1 + 2 * torch.rand_like(batch[:, f])
#     out_f, pre_acts_f, _ = model(batch_f)
#     embed_f = pre_acts_f["layers.0.input_layer.weight"]
#     labels_f = embed_f + torch.relu(model.W_E[f] * 1.0)
#     mse_f = F.mse_loss(out_f, labels_f).item()
#     print(f"MSE for feature {f}: {mse_f}")
#     dot_products_f = einops.einsum(out_f - embed_f, model.W_E, "b d, i d -> b i")
#     plt.scatter(
#         batch_f[:].flatten().cpu().detach(),
#         dot_products_f[:].flatten().cpu().detach(),
#         marker=".",
#         s=4,
#     )
# plt.show()
# fig, axes = plt.subplots(model.n_features, 1, figsize=(10, 40), constrained_layout=True)
# for f in range(model.n_features):
#     batch_f = batch.clone()
#     batch_f[:, f] = -1 + 2 * torch.rand_like(batch[:, f])
#     out_f, pre_acts_f, _ = model(batch_f)
#     embed_f = pre_acts_f["layers.0.input_layer.weight"]
#     labels_f = embed_f + torch.relu(model.W_E[f] * 1.0)
#     mse_f = F.mse_loss(out_f, labels_f).item()
#     print(f"MSE for feature {f}: {mse_f}")
#     dot_products_f = einops.einsum(out_f - embed_f, model.W_E, "b d, i d -> b i")
#     axes[f].set_title(f"Feature {f}, MSE: {mse_f:.4f}")
#     axes[f].scatter(
#         batch_f[:].flatten().cpu().detach(),
#         dot_products_f[:].flatten().cpu().detach(),
#         marker=".",
#         s=4,
#     )
# plt.show()
# # %%
# # Lets run the following eval: For all pairs of features, measure MSE if just those features are
# # active.
# batch, labels = dataset.generate_batch(task_config["batch_size"])
# mse_losses = torch.zeros(model.n_features, model.n_features)
# for f1 in range(model.n_features):
#     for f2 in range(model.n_features):
#         batch_f1f2 = torch.zeros_like(batch)
#         batch_f1f2[:, f1] = -1 + 2 * torch.rand_like(batch[:, f1])
#         batch_f1f2[:, f2] = -1 + 2 * torch.rand_like(batch[:, f2])
#         out_f1f2, pre_acts_f1f2, _ = model(batch_f1f2)
#         embed_f1f2 = pre_acts_f1f2["layers.0.input_layer.weight"]
#         labels_f1f2 = (
#             embed_f1f2 + torch.relu(model.W_E[f1] * 1.0) + torch.relu(model.W_E[f2] * 1.0)
#             if f1 != f2
#             else embed_f1f2 + torch.relu(model.W_E[f1] * 1.0)
#         )
#         mse_f1f2 = F.mse_loss(out_f1f2, labels_f1f2).item()
#         print(f"MSE for features {f1} and {f2}: {mse_f1f2}")
#         mse_losses[f1, f2] = mse_f1f2

# plt.imshow(mse_losses.cpu().detach())
# for i in range(model.n_features):
#     for j in range(model.n_features):
#         plt.text(i, j, f"{mse_losses[i, j]:.4f}", ha="center", va="center", color="w")
# plt.title("MSE for each pair of features")
# plt.xlabel("Feature 2")
# plt.ylabel("Feature 1")
# plt.colorbar()
# plt.show()

# # %%
# # For each feature, plot the max cosine sim with other features
# cos_sims = einops.einsum(model.W_E, model.W_E, "i d, j d -> i j")
# plt.imshow(cos_sims.sort(dim=1, descending=True).values.cpu().detach())
# plt.title("Cosine similarity between features")
# plt.xlabel("Most similar features (sorted)")
# plt.ylabel("Feature")
# plt.colorbar()
# plt.show()
# # For each feature, plot the max cosine sim with other features
# cos_sims = einops.einsum(model.W_E, model.W_E, "i d, j d -> i j")
# plt.imshow(cos_sims.cpu().detach())
# plt.title("Cosine similarity between features")
# plt.xlabel("Most similar features (unsorted)")
# plt.ylabel("Feature")
# plt.colorbar()
# plt.show()
# # %%

# f = 0  # blue and purple, 4
# batch, labels = dataset.generate_batch(task_config["batch_size"])
# batch_green = torch.zeros_like(batch)
# batch_green[:, f] = -1 + 2 * torch.rand_like(batch[:, f])
# out_green, pre_acts_green, _ = model(batch_green)
# embed_green = pre_acts_green["layers.0.input_layer.weight"]
# labels_green = embed_green + torch.relu(model.W_E[f] * 1.0)
# mse_green = F.mse_loss(out_green, labels_green).item()
# print(f"MSE for feature {f}: {mse_green}")

# # Read-off of each feature from the output
# for f in range(model.n_features):
#     # Dot product of the output with the feature vector
#     dot_product = einops.einsum(out_green, model.W_E[f], "b d, d -> b")
#     print(f"Feature {f}: {dot_product.mean().item()}")

# # %%
# W_E: Float[Tensor, "n_features d_embed"]
# print("Blue:", W_E[0])
# print("Purple:", W_E[4])
# print("Blue outputs:")
# mlp_out_green = out_green - embed_green
# for i in range(10):
#     if mlp_out_green[i].abs().max() > 0:
#         print(mlp_out_green[i, :].detach().cpu().numpy())
# print(f"MSE for feature {f}: {mse_green}")
# %%

# If we look at inputs that have multiple features active, and we apply d_embed at the end, and look
# at readoffs
# %%
batch, labels = dataset.generate_batch(task_config["batch_size"])
for f in range(model.n_features):
    batch = torch.zeros_like(batch)
    batch[:, f] = -1 + 2 * 0.75
    out, pre_acts, _ = model(batch)
    embed = pre_acts["layers.0.input_layer.weight"]
    mlp_out = out - embed
    feature_out = einops.einsum(
        mlp_out, model.W_E, "batch d_embed, n_features d_embed  -> batch n_features"
    )
    print(feature_out.shape)
    cmap_viridis = plt.get_cmap("viridis")
    color = cmap_viridis(f / model.n_features)
    plt.plot(
        feature_out[0, :].detach().cpu().numpy() / batch[0, f].detach().cpu().numpy(),
        color=color,
    )
plt.show()

# %% Now for 2 features:

batch, labels = dataset.generate_batch(task_config["batch_size"])
SNR = torch.zeros(model.n_features, model.n_features)
for f1 in tqdm(range(model.n_features)):
    for f2 in range(model.n_features):
        batch = torch.zeros_like(batch)
        batch[:, f1] = -1 + 2 * 0.75  # torch.rand_like(batch[:, f1])
        batch[:, f2] = -1 + 2 * 0.75  # torch.rand_like(batch[:, f2])
        out, pre_acts, _ = model(batch)
        embed = pre_acts["layers.0.input_layer.weight"]
        mlp_out = out - embed
        feature_out = einops.einsum(
            mlp_out, model.W_E, "batch d_embed, n_features d_embed  -> batch n_features"
        )
        signal = min(feature_out[0, f1].abs(), feature_out[0, f2].abs())
        noise = feature_out[0, :].std()
        SNR[f1, f2] = signal / noise
        plt.plot(feature_out[0, :].detach().cpu().numpy(), color=color)
        plt.show()

plt.imshow(SNR.cpu().detach())
plt.colorbar()
plt.title("SNR for each pair of features")
plt.show()
# %% Now for 1 on 1 off

batch, labels = dataset.generate_batch(task_config["batch_size"])
SNR = torch.zeros(model.n_features, model.n_features)
for f1 in tqdm(range(model.n_features)):
    for f2 in range(model.n_features):
        if f1 == f2:
            continue
        batch = torch.zeros_like(batch)
        batch[:, f1] = -1 + 2 * 0.75  # torch.rand_like(batch[:, f1])
        batch[:, f2] = -1 + 2 * 0.25  # torch.rand_like(batch[:, f2])
        out, pre_acts, _ = model(batch)
        embed = pre_acts["layers.0.input_layer.weight"]
        mlp_out = out - embed
        feature_out = einops.einsum(
            mlp_out, model.W_E, "batch d_embed, n_features d_embed  -> batch n_features"
        )
        signal = feature_out[0, f1]
        noise = feature_out[0, :].std()
        SNR[f1, f2] = signal / noise
        plt.plot(feature_out[0, :].detach().cpu().numpy(), color=color)
        plt.show()

plt.imshow(SNR.cpu().detach())
plt.colorbar()
plt.title("SNR for each pair of features")
plt.show()
# %%

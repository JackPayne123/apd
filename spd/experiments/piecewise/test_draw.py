# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import einsum

# %%
m = torch.load("out/ml_topk0.625_topkrecon0.1_topkl2_1.0_lr0.01_bs2048lay2_/model_50000.pth")
m = torch.load("out/ml_topk0.3_topkrecon0.1_topkl2_1.0_lr0.01_bs2000lay2_/model_50000.pth")

# m["mlps.0.linear1.A"]
mlp_components = []
mlps = []
for i in range(2):
    print(m[f"mlps.{i}.linear1.A"].shape)
    print(m[f"mlps.{i}.linear1.B"].shape)
    W_in = einsum(
        m[f"mlps.{i}.linear1.A"], m[f"mlps.{i}.linear1.B"], "embed k, k mlp -> k embed mlp"
    )
    W_out = einsum(
        m[f"mlps.{i}.linear2.A"], m[f"mlps.{i}.linear2.B"], "mlp k, k embed -> k mlp embed"
    )
    mlp_components.append((W_in, W_out))
    mlps.append((W_in.sum(dim=0), W_out.sum(dim=0)))


# %%
fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
# Remove axes and labels

n_layers = len(mlps)
d_embed = mlps[0][0].shape[0]
d_mlp = mlps[0][0].shape[1]


def plot_nodes(ax):
    # Resid
    for i in range(n_layers + 1):
        for j in range(d_embed):
            # edgecolor black, fill grey
            ax.scatter(i, j, color="lightgrey", edgecolor="black", s=30)
    # MLP
    for i in range(n_layers):
        for j in range(d_mlp):
            ax.scatter(i + 0.5, d_embed + j, color="lightgrey", edgecolor="black", s=30)


lw_scale = 0.1

plot_nodes(ax)
# Weights
for i, mlp in enumerate(mlps):
    W_in, W_out = mlp
    for j in range(d_embed):
        for k in range(d_mlp):
            ax.plot(
                [i, i + 0.5],
                [j, d_embed + k],
                lw=lw_scale * W_in.abs()[j, k],
                color="r" if W_in[j, k] > 0 else "b",
            )
            ax.plot(
                [i + 0.5, i + 1],
                [d_embed + k, j],
                lw=lw_scale * W_out.abs()[k, j],
                color="r" if W_out[k, j] > 0 else "b",
            )


# %%

# Plot indiv components
n_comp = mlp_components[0][0].shape[0]
n_rows = n_comp
n_cols = 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5 * n_comp), constrained_layout=True)
for c in range(n_comp):
    ax = axes[c]
    plot_nodes(ax)
    for i, mlp in enumerate(mlp_components):
        W_in, W_out = mlp
        W_in = W_in[c]
        W_out = W_out[c]
        for j in range(d_embed):
            for k in range(d_mlp):
                ax.plot(
                    [i, i + 0.5],
                    [j, d_embed + k],
                    lw=lw_scale * W_in.abs()[j, k],
                    color="r" if W_in[j, k] > 0 else "b",
                )
                ax.plot(
                    [i + 0.5, i + 1],
                    [d_embed + k, j],
                    lw=lw_scale * W_out.abs()[k, j],
                    color="r" if W_out[k, j] > 0 else "b",
                )

# %%

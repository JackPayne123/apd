# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDTransformer,
)
from spd.experiments.piecewise.piecewise_decomposition import get_model_and_dataloader

# plot_subnetwork_correlations,
from spd.models.components import ParamComponents, ParamComponentsFullRank
from spd.run_spd import (
    Config,
    PiecewiseConfig,
)
from spd.utils import REPO_ROOT


def plot_single_network(ax: plt.Axes, weights: list[dict[str, Float[Tensor, "i j"]]]):
    n_layers = len(weights)
    d_embed = weights[0]["W_in"].shape[0]
    d_mlp = weights[0]["W_in"].shape[1]
    assert all(W["W_in"].shape == (d_embed, d_mlp) for W in weights)
    assert all(W["W_out"].shape == (d_mlp, d_embed) for W in weights)

    # Define node positions
    x_embed = np.linspace(0.05, 0.45, d_embed)
    x_mlp = np.linspace(0.55, 0.95, d_mlp)

    # Plot nodes
    for lay in range(n_layers + 1):
        ax.scatter(x_embed, [2 * lay] * d_embed, s=100, color="grey", edgecolors="k", zorder=3)
    for lay in range(n_layers):
        ax.scatter(x_mlp, [2 * lay + 1] * d_mlp, s=100, color="grey", edgecolors="k", zorder=3)

    # Plot edges
    cmap = plt.get_cmap("RdBu_r")
    for lay in range(n_layers):
        # Normalize weights
        W_in_norm = weights[lay]["W_in"] / weights[lay]["W_in"].abs().max()
        W_out_norm = weights[lay]["W_out"] / weights[lay]["W_out"].abs().max()
        for i in range(d_embed):
            for j in range(d_mlp):
                weight = W_in_norm[i, j].item()
                color = cmap(0.5 * (weight + 1))
                ax.plot(
                    [x_embed[i], x_mlp[j]],
                    [2 * lay + 2, 2 * lay + 1],
                    color=color,
                    linewidth=abs(weight),
                )
                weight = W_out_norm[j, i].item()
                color = cmap(0.5 * (weight + 1))
                ax.plot(
                    [x_mlp[j], x_embed[i]],
                    [2 * lay + 1, 2 * lay],
                    color=color,
                    linewidth=abs(weight),
                )

    ax.axis("off")
    ax.set_xlabel("Output")


def get_weight(general_param_components: ParamComponents | ParamComponentsFullRank):
    if isinstance(general_param_components, ParamComponentsFullRank):
        weight: Float[Tensor, "k i j"] = general_param_components.subnetwork_params
        return weight
    elif isinstance(general_param_components, ParamComponents):
        a: Float[Tensor, "i k"] = general_param_components.A
        b: Float[Tensor, "k j"] = general_param_components.B
        weight: Float[Tensor, "k i j"] = einsum(a, b, "i k, k j -> k i j")
        return weight
    else:
        raise ValueError(f"Unknown type: {type(general_param_components)}")


def plot_resnet(model: PiecewiseFunctionSPDTransformer | PiecewiseFunctionSPDFullRankTransformer):
    n_components = model.k
    mlps: torch.nn.ModuleList = model.mlps
    n_layers = len(mlps)

    subnetworks = {}
    for k in range(n_components):
        subnetworks[k] = []
        for lay in range(n_layers):
            W_in = get_weight(mlps[lay].linear1)
            W_out = get_weight(mlps[lay].linear2)
            subnetworks[k].append({"W_in": W_in[k], "W_out": W_out[k]})

    fig, axs = plt.subplots(
        1,
        n_components + 1,
        figsize=(3 * (n_components + 1), 4 * n_layers),
        constrained_layout=True,
    )
    axs = np.array(axs)
    for k in range(n_components):
        plot_single_network(axs[k + 1], subnetworks[k])

    return fig


# Load and process the model
pretrained_path = (
    REPO_ROOT
    / "spd/experiments/piecewise/out/plot4sn2l_seed0_topk2.49e-01_topkrecon1.00e+00_topkl2_1.00e+00_lr1.00e-02_bs2000lay2/model_30000.pth"
)
with open(pretrained_path.parent / "config.json") as f:
    config_dict = json.load(f)
    config = Config(**config_dict)

device = "cuda" if torch.cuda.is_available() else "cpu"

assert isinstance(config.task_config, PiecewiseConfig)

hardcoded_model, spd_model, dataloader, test_dataloader = get_model_and_dataloader(
    config, device, out_dir=None
)

fig = plot_resnet(spd_model)
plt.show()

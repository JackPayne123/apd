# %%

import json

import matplotlib.collections as mc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.run_spd import (
    Config,
)
from spd.utils import REPO_ROOT

# %%
pretrained_path = (
    REPO_ROOT
    # / "spd/experiments/tms/out/fr_topk2.50e-01_topkrecon1.00e+01_topkl2_1.00e+00_lr1.00e-03_bs2048_ft5_hid2/model_30000.pth"
    / "spd/experiments/tms/demo_spd_model/model_30000.pth"
)

with open(pretrained_path.parent / "config.json") as f:
    config_dict = json.load(f)
    config = Config(**config_dict)

assert config.full_rank, "This script only works for full rank models"
subnet = torch.load(pretrained_path, map_location="cpu")["subnetwork_params"]


# %%
def plot_vectors(
    subnet: Float[Tensor, "n_instances n_subnets n_features n_hidden"], n_instances: int = 5
) -> plt.Figure:
    """2D polygon plot of each subnetwork.

    Adapted from
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb.
    """
    n_data_instances, n_subnets, n_features, n_hidden = subnet.shape
    assert (
        n_instances <= n_data_instances
    ), "n_instances must be less than or equal to n_data_instances"
    sel = range(n_instances)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color",
        plt.cm.viridis(np.array([1.0])),  # type: ignore
    )
    plt.rcParams["figure.dpi"] = 200
    fig, axs = plt.subplots(
        n_subnets + 1, len(sel) + 1, figsize=(2 * (len(sel) + 1), 2 * (n_subnets + 1))
    )
    axs = np.array(axs)
    for j in range(n_subnets + 1):
        # Add a new column for labels
        label_ax = axs[j, 0]
        label_ax.axis("off")
        label = "Sum of subnets" if j == 0 else f"Subnet {j-1}"
        label_ax.text(
            0.5, 0.5, label, rotation=0, va="center", ha="center", transform=label_ax.transAxes
        )

        for i, ax in enumerate(axs[j, 1:]):
            if j == 0:
                # First, plot the addition of the subnetworks
                arr = subnet[i].sum(dim=0).cpu().detach().numpy()
            else:
                # Plot the jth subnet
                arr = subnet[i, j - 1].cpu().detach().numpy()
            colors = [mcolors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
            ax.scatter(arr[:, 0], arr[:, 1], c=colors[0 : len(arr[:, 0])])
            ax.set_aspect("equal")
            ax.add_collection(
                mc.LineCollection(np.stack((np.zeros_like(arr), arr), axis=1), colors=colors)  # type: ignore
            )

            z = 1.5
            ax.set_facecolor("#FCFBF8")
            ax.set_xlim((-z, z))
            ax.set_ylim((-z, z))
            ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_position("center")

            if j == n_subnets:
                ax.set_xlabel(f"Instance {i}", rotation=0, ha="center", labelpad=60)

    return fig


fig = plot_vectors(subnet)
fig.savefig(pretrained_path.parent / "polygon_diagram.png", bbox_inches="tight")

# %%

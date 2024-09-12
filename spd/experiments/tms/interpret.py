# %%

import json

import matplotlib.collections as mc
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
pretrained_path = REPO_ROOT / "spd/experiments/tms/demo_spd_model/model_30000.pth"

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
    color = plt.cm.viridis(np.array([0.0]))  # type: ignore

    fig, axs = plt.subplots(len(sel), n_subnets + 1, figsize=(2 * (n_subnets + 1), 2 * (len(sel))))
    axs = np.array(axs)
    for j in range(n_subnets + 1):
        for i, ax in enumerate(axs[:, j]):
            if j == 0:
                # First, plot the addition of the subnetworks
                arr = subnet[i].sum(dim=0).cpu().detach().numpy()
            else:
                # Plot the jth subnet
                arr = subnet[i, j - 1].cpu().detach().numpy()
            ax.scatter(arr[:, 0], arr[:, 1], c=color)
            ax.set_aspect("equal")
            ax.add_collection(
                mc.LineCollection(np.stack((np.zeros_like(arr), arr), axis=1), colors=[color])  # type: ignore
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

            if i == len(sel) - 1:
                label = "Sum of subnets" if j == 0 else f"Subnet {j-1}"
                ax.set_xlabel(label, rotation=0, ha="center", labelpad=60)
            if j == 0:
                ax.set_ylabel(f"Instance {i}", rotation=90, ha="center", labelpad=60)

    return fig


fig = plot_vectors(subnet)
fig.savefig(pretrained_path.parent / "polygon_diagram.png", bbox_inches="tight", dpi=200)
print(f"Saved figure to {pretrained_path.parent / 'polygon_diagram.png'}")

# %%

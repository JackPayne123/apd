import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.models.base import SPDModel
from spd.module_utils import collect_nested_module_attrs


def plot_subnetwork_attributions_statistics(
    mask: Float[Tensor, "batch_size n_instances m"],
) -> dict[str, plt.Figure]:
    """Plot vertical bar charts of the number of active subnetworks over the batch for each instance."""
    batch_size = mask.shape[0]
    if mask.ndim == 2:
        n_instances = 1
        mask = einops.repeat(mask, "batch m -> batch n_instances m", n_instances=1)
    else:
        n_instances = mask.shape[1]

    fig, axs = plt.subplots(
        ncols=n_instances, nrows=1, figsize=(5 * n_instances, 5), constrained_layout=True
    )

    axs = np.array([axs]) if n_instances == 1 else np.array(axs)
    for i, ax in enumerate(axs):
        values = mask[:, i].sum(dim=1).cpu().detach().numpy()
        bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
        counts, _ = np.histogram(values, bins=bins)
        bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
        ax.set_xticks(bins[:-1])
        ax.set_xticklabels([str(b) for b in bins[:-1]])

        # Only add y-label to first subplot
        if i == 0:
            ax.set_ylabel("Count")

        ax.set_xlabel("Number of active subnetworks")
        ax.set_title(f"Instance {i+1}")

        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.suptitle(f"Active subnetworks on current batch (batch_size={batch_size})")
    return {"subnetwork_attributions_statistics": fig}


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.1f",
    norm: plt.Normalize | None = None,
) -> None:
    # Useful to have bigger text for small matrices
    fontsize = 8 if matrix.numel() < 50 else 4
    norm = norm if norm is not None else CenteredNorm()
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=norm)
    # If less than 500 elements, show the values
    if matrix.numel() < 500:
        for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
            ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=fontsize)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])


def plot_As(model: SPDModel, device: str) -> plt.Figure:
    """Plot the A matrices for each instance."""
    # Collect all A matrices
    As = collect_nested_module_attrs(model, attr_name="A", include_attr_name=False)
    n_instances = model.n_instances

    # Create figure for plotting
    fig, axs = plt.subplots(
        len(As),
        n_instances,
        figsize=(5 * n_instances, 5 * len(As)),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot each A matrix for each instance
    for i in range(n_instances):
        axs[0, i].set_title(f"Instance {i}")
        for j, (A_name, A) in enumerate(As.items()):
            # A has shape (n_instances, d_in, m)
            A_data = A[i].detach().cpu().numpy()
            im = axs[j, i].matshow(A_data, aspect="auto", cmap="coolwarm")
            if i == 0:
                axs[j, i].set_ylabel("d_in index")
            axs[j, i].set_xlabel("Component index")
            axs[j, i].set_title(A_name)
            images.append(im)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(A.min().item() for A in As.values()),
        vmax=max(A.max().item() for A in As.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    return fig

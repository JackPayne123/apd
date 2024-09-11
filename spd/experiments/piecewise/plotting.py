from pathlib import Path

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from spd.experiments.piecewise.models import PiecewiseFunctionSPDTransformer
from spd.utils import calc_attributions_rank_one


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.0f",
) -> None:
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=CenteredNorm())
    for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
        ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=4)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])


def plot_components_fullrank(
    model: PiecewiseFunctionSPDTransformer,
    step: int,
    out_dir: Path | None,
    **_,
) -> dict[str, plt.Figure]:
    # Not implemented attribution score plots, or multi-layer plots, yet.
    assert model.n_layers == 1
    ncols = 2
    nrows = model.k + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(16 * ncols, 3 * nrows), constrained_layout=True)
    assert isinstance(axs, np.ndarray)
    plot_matrix(
        axs[0, 0],
        einops.einsum(model.mlps[0].linear1.subnetwork_params, "k ... -> ..."),
        "W_in, sum over k",
        "Neuron index",
        "Embedding index",
    )
    plot_matrix(
        axs[0, 1],
        einops.einsum(model.mlps[0].linear2.subnetwork_params, "k ... -> ...").T,
        "W_out.T, sum over k",
        "Neuron index",
        "",
    )

    for k in range(model.k):
        mlp = model.mlps[0]
        W_in_k = mlp.linear1.subnetwork_params[k]
        ax = axs[k + 1, 0]  # type: ignore
        plot_matrix(ax, W_in_k, f"W_in_k, k={k}", "Neuron index", "Embedding index")
        W_out_k = mlp.linear2.subnetwork_params[k].T
        ax = axs[k + 1, 1]  # type: ignore
        plot_matrix(ax, W_out_k, f"W_out_k.T, k={k}", "Neuron index", "")
    if out_dir is not None:
        fig.savefig(out_dir / f"matrices_l0_s{step}.png", dpi=300)
        print(f"saved to {out_dir / f'matrices_l0_s{step}.png'}")
    return {"matrices_l0_s{step}": fig}


def plot_components(
    model: PiecewiseFunctionSPDTransformer,
    step: int,
    out_dir: Path | None,
    device: str,
    slow_images: bool,
    **_,
) -> dict[str, plt.Figure]:
    # Create a batch of inputs with different control bits active
    x_val = torch.tensor(2.5, device=device)
    batch_size = model.n_inputs - 1  # Assuming first input is for x_val and rest are control bits
    x = torch.zeros(batch_size, model.n_inputs, device=device)
    x[:, 0] = x_val
    x[torch.arange(batch_size), torch.arange(1, batch_size + 1)] = 1
    # Forward pass to get the output and inner activations
    out, layer_acts, inner_acts = model(x)
    # Calculate attribution scores
    attribution_scores = calc_attributions_rank_one(out=out, inner_acts=inner_acts)
    attribution_scores_normed = attribution_scores / attribution_scores.std(dim=1, keepdim=True)
    # Get As and Bs and ABs
    n_layers = model.n_layers
    assert len(model.all_As()) == len(model.all_Bs()), "A and B matrices must have the same length"
    assert len(model.all_As()) % 2 == 0, "A and B matrices must have an even length (MLP in + out)"
    assert len(model.all_As()) // 2 == n_layers, "Number of A and B matrices must be 2*n_layers"
    As = model.all_As()
    Bs = model.all_Bs()
    ABs = [torch.einsum("...fk,...kg->...fg", As[i], Bs[i]) for i in range(len(As))]
    ABs_by_k = [torch.einsum("...fk,...kg->...kfg", As[i], Bs[i]) for i in range(len(As))]

    # Figure for attribution scores
    fig_a, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    fig_a.suptitle(f"Subnetwork Analysis (Step {step})")
    plot_matrix(
        ax,
        attribution_scores_normed,
        "Normalized attribution Scores",
        "Subnetwork index",
        "Function index",
    )
    if out_dir:
        fig_a.savefig(out_dir / f"attribution_scores_s{step}.png", dpi=300)
        plt.close(fig_a)
        tqdm.write(f"Saved attribution scores to {out_dir / f'attribution_scores_s{step}.png'}")

    # Figures for A, B, AB of each layer
    n_rows = 3 + model.k if slow_images else 3
    n_cols = 4
    figsize = (8 * n_cols, 4 + 4 * n_rows)
    figs = [plt.figure(figsize=figsize, constrained_layout=True) for _ in range(n_layers)]
    # Plot normalized attribution scores

    for n in range(n_layers):
        fig = figs[n]
        gs = fig.add_gridspec(n_rows, n_cols)
        plot_matrix(
            fig.add_subplot(gs[0, 0]),
            As[2 * n],
            f"A (W_in, layer {n})",
            "Subnetwork index",
            "Embedding index",
            "%.1f",
        )
        plot_matrix(
            fig.add_subplot(gs[0, 1:]),
            Bs[2 * n],
            f"B (W_in, layer {n})",
            "Neuron index",
            "Subnetwork index",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[1, 0]),
            Bs[2 * n + 1].T,
            f"B (W_out, layer {n})",
            "Subnetwork index",
            "Embedding index",
            "%.1f",
        )
        plot_matrix(
            fig.add_subplot(gs[1, 1:]),
            As[2 * n + 1].T,
            f"A (W_out, layer {n})",
            "Neuron index",
            "",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[2, :2]),
            ABs[2 * n],
            f"AB summed (W_in, layer {n})",
            "Neuron index",
            "Embedding index",
            "%.2f",
        )
        plot_matrix(
            fig.add_subplot(gs[2, 2:]),
            ABs[2 * n + 1].T,
            f"AB.T  summed (W_out.T, layer {n})",
            "Neuron index",
            "",
            "%.2f",
        )
        if slow_images:
            for k in range(model.k):
                plot_matrix(
                    fig.add_subplot(gs[3 + k, :2]),
                    ABs_by_k[2 * n][k],
                    f"AB k={k} (W_in, layer {n})",
                    "Neuron index",
                    "Embedding index",
                    "%.2f",
                )
                plot_matrix(
                    fig.add_subplot(gs[3 + k, 2:]),
                    ABs_by_k[2 * n + 1][k].T,
                    f"AB.T k={k} (W_out.T, layer {n})",
                    "Neuron index",
                    "Embedding index",
                    "%.2f",
                )

        if out_dir:
            fig.savefig(out_dir / f"matrices_l{n}_s{step}.png", dpi=300)
            plt.close(fig_a)
            tqdm.write(f"Saved matrix analysis to {out_dir / f'matrices_l{n}_s{step}.png'}")

    return {"attrib_scores": fig_a, **{f"matrices_l{n}_s{step}": fig for n, fig in enumerate(figs)}}

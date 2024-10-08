"""Run SPD on a model."""

from collections.abc import Callable
from itertools import cycle
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.bigrams.model import BigramModel
from spd.experiments.piecewise.models import PiecewiseFunctionTransformer
from spd.models.base import Model, SPDFullRankModel, SPDModel
from spd.run_spd import (
    Config,
    get_lr_schedule_fn,
    get_lr_with_warmup,
    get_sparsity_coeff_linear_warmup,
    get_step_pnorm,
)
from spd.utils import (
    calc_topk_mask,
)


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.0f",
) -> None:
    # Useful to have bigger text for small matrices
    fontsize = 8 if matrix.numel() < 50 else 4
    matrix = np.atleast_2d(matrix.detach().cpu().numpy())  # type: ignore
    im = ax.matshow(matrix, cmap="coolwarm", norm=CenteredNorm())
    # for (j, i), label in np.ndenumerate(matrix):
    #     ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=fontsize)
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


def plot_param_dict(
    param_dict: dict[str, torch.Tensor],
    prefix: str = "",
) -> dict[str, plt.Figure]:
    # n_params = len(param_dict)
    fig_dict = {}
    for key, param in param_dict.items():
        k = param.shape[0]
        fig, axes = plt.subplots(nrows=k + 1, ncols=1, figsize=(8, 6 * k), constrained_layout=True)
        axes = np.atleast_1d(axes)  # type: ignore
        fig.suptitle(f"{prefix}{key}")
        plot_matrix(axes[0], param.sum(dim=0), "", "", "")
        axes[0].set_ylabel("Sum")
        for ki in range(k):
            plot_matrix(axes[ki + 1], param[ki], key, "x", "y")
            axes[ki + 1].set_ylabel(f"k={ki}")
        fig_dict[f"{prefix}{key}"] = fig
    return fig_dict


def calc_param_match_loss(
    k_params: dict[str, Float[Tensor, " k ... d"]],
    pretrained_params: dict[str, Float[Tensor, " ... d"]],
    device: str,
) -> Float[Tensor, ""]:
    param_match_loss = torch.tensor(0.0, device=device)
    for key, value in k_params.items():
        param_match_loss += (value.sum(dim=0) - pretrained_params[key]).pow(2).mean()
    return param_match_loss / len(k_params)


# TODO: orthog_loss


def calc_recon_loss(
    topk_out: Float[Tensor, " batch n_outputs"],
    pretrained_out: Float[Tensor, " batch n_outputs"],
) -> Float[Tensor, ""]:
    return (topk_out - pretrained_out).pow(2).mean(dim=-1).mean(dim=0)


def calc_topk_l2_loss(
    topk_mask: Float[Tensor, " batch k"],
    k_params: dict[str, Float[Tensor, " k ... d"]],
) -> Float[Tensor, ""]:
    l2_loss = torch.tensor(0.0, device=topk_mask.device)
    for _, value in k_params.items():
        l2_loss += (
            einops.einsum(
                topk_mask,
                value,
                "batch k, k ... d -> batch ... d",
            )
            .pow(2)
            .mean()
        )
    return l2_loss / len(k_params)


def calc_lp_loss(attribs: Float[Tensor, "batch k"], p: float) -> Float[Tensor, ""]:
    return ((attribs.abs() + 1e-16) ** (p * 0.5)).sum(dim=-1).mean(dim=0)


def calc_topk_param_attrib_loss(
    k_params: dict[str, Float[Tensor, " k ... d"]],
    topk_mask: Float[Tensor, " batch k"],
    pretrained_params: dict[str, Float[Tensor, " ... d"]],
    pretrained_param_grads: dict[str, Float[Tensor, " batch ... d"]],
    batch_size: int,
    device: str,
) -> Float[Tensor, ""]:
    loss: Float[Tensor, " batch"] = torch.zeros(batch_size, device=device)
    for key, value in k_params.items():
        param_diff: Float[Tensor, " ... d"] = (
            einops.einsum(topk_mask, value, "batch k, k ... d -> batch ... d")
            - pretrained_params[key]
        )
        param_grad: Float[Tensor, " batch ... d"] = pretrained_param_grads[key]
        param_attrib = param_diff * param_grad
        loss += einops.reduce(param_attrib, "batch ... d -> batch", "mean")
    return (loss**2).mean(dim=0)


def optimize(
    model: SPDModel | SPDFullRankModel | None,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    pretrained_model: Model | None,
    param_map: dict[str, str] | None = None,
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    assert not hasattr(model, "n_instances"), "Instances not supported"
    assert config.full_rank, "Only full rank supported"
    assert not config.unit_norm_matrices, "Unit norm matrices not supported"
    assert pretrained_model is not None, "Only works given a pretrained model"
    assert config.out_recon_coeff is None, "We're (currently) not supporting out_recon_coeff"

    # We don't need no SPDModel. We just take the state dict of the pretrained model and duplicate
    # it k times. Just optimize this dictionary of parameters.
    pretrained_model.to(device=device)
    pretrained_params = pretrained_model.state_dict()
    if isinstance(pretrained_model, BigramModel):
        decomposable_params = pretrained_params
    elif isinstance(pretrained_model, PiecewiseFunctionTransformer):
        decomposable_params = [
            "mlps.0.input_layer.weight",
            "mlps.0.output_layer.weight",
            "mlps.0.input_layer.bias",
        ]
    else:
        raise ValueError(f"Unsure how to decompose {pretrained_model}")
    k_params = {}
    k = config.task_config.k
    assert config.topk is not None, "Need topk"
    assert k is not None, "Need k"
    for key, value in pretrained_params.items():
        if key in decomposable_params:
            shape = [k, *value.shape]
            k_params[key] = torch.empty(shape, device=device, dtype=value.dtype, requires_grad=True)

    # Okay now for piecewise I want to use the handcoded SPD initialization so I am gonna use the
    # SPD model after all, but literally only to get the initialization params. Also, it's annoying
    # that SPD model uses different parameter names so we're using this map. Also, the params are
    # transposed.
    if config.initialize_spd == "oldSPD":
        assert isinstance(pretrained_model, PiecewiseFunctionTransformer)
        model.to(device=device)
        spd_params = model.state_dict()
        spd_param_map = [
            ("mlps.0.input_layer.weight", "mlps.0.linear1.subnetwork_params", True),
            ("mlps.0.output_layer.weight", "mlps.0.linear2.subnetwork_params", True),
            ("mlps.0.input_layer.bias", "mlps.0.linear1.bias", False),
        ]
        for key, spd_key, transpose in spd_param_map:
            if transpose:
                k_params[key].data[:] = spd_params[spd_key].transpose(-2, -1)
            else:
                k_params[key].data[:] = spd_params[spd_key]
            pass
        del model, spd_params, spd_param_map
    elif config.initialize_spd == "fullcopies":
        for key in k_params:
            k_params[key].data[:] = (
                pretrained_params[key].data + torch.randn_like(k_params[key].data) * 1e0
            )
            pass
    elif config.initialize_spd == "xavier":
        for key in decomposable_params:
            torch.nn.init.xavier_uniform_(k_params[key])
    else:
        raise ValueError(f"Invalid initialize_spd: {config.initialize_spd}")

    def model_func(
        single_mask: Float[Tensor, " k"],
        single_batch: Float[Tensor, " n_inputs"],
        k_params: dict[str, Float[Tensor, " k ..."]],
    ) -> Float[Tensor, " n_outputs"]:
        summed_params = {
            k: einops.einsum(v, single_mask, "k ..., k -> ...") for k, v in k_params.items()
        }
        return functional_call(pretrained_model, summed_params, single_batch)

    def model_func_sq(
        mask: Float[Tensor, "k"],
        single_batch: Float[Tensor, " n_inputs"],
        k_params: dict[str, Float[Tensor, " k ..."]],
    ) -> Float[Tensor, ""]:
        return model_func(mask, single_batch, k_params).pow(2).mean(dim=-1)

    def single_attrib(
        single_batch: Float[Tensor, " n_inputs"],
        k_params: dict[str, Float[Tensor, " k ..."]],
    ) -> Float[Tensor, " k"]:
        single_mask = torch.ones(k, device=device)
        return grad(model_func_sq, argnums=0)(single_mask, single_batch, k_params) ** 2

    def model_func_0(  # TODO: Take output dims into account
        single_batch: Float[Tensor, " n_inputs"],
        summed_params: dict[str, Float[Tensor, " k ..."]],
    ) -> Float[Tensor, ""]:
        return functional_call(pretrained_model, summed_params, single_batch)[0]

    def single_param_grads(single_batch: Float[Tensor, " n_inputs"]) -> Float[Tensor, ""]:
        return grad(model_func_0, argnums=1)(single_batch, pretrained_params)

    batched_param_grads = vmap(single_param_grads, in_dims=0)
    batched_model_func = vmap(model_func, in_dims=(0, 0, None))
    batched_attrib = vmap(single_attrib, in_dims=(0, None))

    # TODO: Can we do torch.compile here?

    # Note that we expect weight decay to be problematic for spd
    opt = torch.optim.AdamW(k_params.values(), lr=config.lr, weight_decay=0.0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    step_lp_sparsity_coeff = None
    step_topk_recon_coeff = None
    total_samples = 0
    data_iter = cycle(dataloader)  # automatically cycles through dataset

    for step in tqdm(range(config.steps + 1), ncols=0):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        for group in opt.param_groups:
            group["lr"] = step_lr

        step_pnorm = config.pnorm or get_step_pnorm(step, config.steps, config.pnorm_end)

        opt.zero_grad(set_to_none=True)
        batch, _ = next(data_iter)
        batch = batch.to(device=device)
        labels = pretrained_model(batch)

        total_samples += batch.shape[0]

        if config.topk_recon_coeff is not None:
            step_topk_recon_coeff = get_sparsity_coeff_linear_warmup(
                step=step,
                steps=config.steps,
                max_sparsity_coeff=config.topk_recon_coeff,
                sparsity_warmup_pct=config.sparsity_warmup_pct,
            )
        if config.lp_sparsity_coeff is not None:
            step_lp_sparsity_coeff = get_sparsity_coeff_linear_warmup(
                step=step,
                steps=config.steps,
                max_sparsity_coeff=config.lp_sparsity_coeff,
                sparsity_warmup_pct=config.sparsity_warmup_pct,
            )

        pretrained_param_grads = batched_param_grads(batch)

        attribs = batched_attrib(batch, k_params)
        topk_mask = calc_topk_mask(attribs, topk=config.topk, batch_topk=config.batch_topk).float()
        topk_out = batched_model_func(topk_mask, batch, k_params)

        topk_param_attrib_loss = calc_topk_param_attrib_loss(
            k_params, topk_mask, pretrained_params, pretrained_param_grads, batch.shape[0], device
        )

        topk_recon_loss = calc_recon_loss(topk_out, labels)

        param_match_loss = calc_param_match_loss(k_params, pretrained_params, device=device)

        topk_l2_loss = calc_topk_l2_loss(topk_mask, k_params)

        lp_sparsity_loss = calc_lp_loss(attribs, step_pnorm)

        orthog_loss = None  # TODO

        # Add up the loss terms
        loss = torch.tensor(0.0, device=device)
        # if config.orthog_coeff is not None:
        #     loss = loss + config.orthog_coeff * orthog_loss.mean()
        if config.param_match_coeff is not None:
            loss = loss + config.param_match_coeff * param_match_loss.mean()
        if step_lp_sparsity_coeff is not None:
            loss = loss + step_lp_sparsity_coeff * lp_sparsity_loss.mean()
        if step_topk_recon_coeff is not None:
            loss = loss + step_topk_recon_coeff * topk_recon_loss.mean()
        if config.topk_l2_coeff is not None:
            loss = loss + config.topk_l2_coeff * topk_l2_loss.mean()
        if config.topk_param_attrib_coeff is not None:
            loss = loss + config.topk_param_attrib_coeff * topk_param_attrib_loss.mean()

        # Logging
        if step % config.print_freq == 0:
            nl = " "
            tqdm.write(f"Step {step}")
            tqdm.write(f"Total loss: {loss.item()}")
            tqdm.write(f"Current pnorm:{nl}{step_pnorm}")
            tqdm.write(f"LP sparsity loss:{nl}{lp_sparsity_loss}")
            tqdm.write(f"Topk recon loss:{nl}{topk_recon_loss}")
            tqdm.write(f"topk l2 loss:{nl}{topk_l2_loss}")
            tqdm.write(f"param match loss:{nl}{param_match_loss}")
            tqdm.write(f"topk param attrib loss:{nl}{topk_param_attrib_loss}")
            # tqdm.write(f"Orthog loss:{nl}{orthog_loss}")
            if config.wandb_project:
                wandb.log(
                    {
                        "pnorm": step_pnorm,
                        "lr": step_lr,
                        "total_loss": loss.mean().item(),
                        "lp_sparsity_loss": lp_sparsity_loss.mean().item(),
                        "topk_recon_loss": topk_recon_loss.mean().item(),
                        "param_match_loss": param_match_loss.mean().item(),
                        "topk_l2_loss": topk_l2_loss.mean().item(),
                        "topk_param_attrib_loss": topk_param_attrib_loss.mean().item(),
                    },
                    step=step,
                )

        if (
            True
            and config.image_freq is not None
            and step % config.image_freq == 0
            and (step > 0 or not config.slow_images)
        ):
            fig_dict = plot_param_dict(k_params)
            # Plot gradients
            grads = {k: v.grad for k, v in k_params.items() if v.grad is not None}
            fig_dict_grads = plot_param_dict(grads, prefix="grad_")
            fig_dict = {**fig_dict, **fig_dict_grads}
            if config.wandb_project:
                wandb.log(
                    {k: wandb.Image(v) for k, v in fig_dict.items()},
                    step=step,
                )
                plt.close("all")
        # if (
        #     config.save_freq is not None
        #     and step % config.save_freq == 0
        #     and step > 0
        #     and out_dir is not None
        # ):
        #     torch.save(model.state_dict(), out_dir / f"model_{step}.pth")
        #     tqdm.write(f"Saved model to {out_dir / f'model_{step}.pth'}")
        #     with open(out_dir / "config.json", "w") as f:
        #         json.dump(config.model_dump(), f, indent=4)
        #     tqdm.write(f"Saved config to {out_dir / 'config.json'}")

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            loss.backward()

            if step % config.print_freq == 0 and config.wandb_project:
                # Calculate gradient norm
                grad_norm: float = 0.0
                for param in k_params.values():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm()  # type: ignore
                wandb.log({"grad_norm": grad_norm}, step=step)

            opt.step()

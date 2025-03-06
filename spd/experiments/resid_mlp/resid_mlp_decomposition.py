"""Residual Linear decomposition script."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config, ResidualMLPTaskConfig
from spd.experiments.resid_mlp.models import (
    ResidualMLPModel,
    ResidualMLPSPDConfig,
    ResidualMLPSPDModel,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.log import logger
from spd.models.components import Gate
from spd.plotting import plot_AB_matrices, plot_mask_vals
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    get_device,
    load_config,
    set_seed,
)
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(
    config: Config,
    n_features: int,
    n_layers: int,
    d_resid: int,
    d_mlp: int,
    m: int | None,
    init_scale: float,
) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"scale{init_scale}_ft{n_features}_lay{n_layers}_resid{d_resid}_mlp{d_mlp}"
    return config.wandb_run_name_prefix + run_suffix


def plot_subnetwork_attributions(
    attribution_scores: Float[Tensor, "batch n_instances m"],
    out_dir: Path | None,
    step: int | None,
) -> plt.Figure:
    """Plot subnetwork attributions."""
    # Plot a row with n_instances
    # Each column is a different instance
    n_instances = attribution_scores.shape[1]
    fig, ax = plt.subplots(
        nrows=1, ncols=n_instances, figsize=(5 * n_instances, 5), constrained_layout=True
    )
    axs = np.array([ax]) if n_instances == 1 else np.array(ax)
    im = None
    for i in range(n_instances):
        im = axs[i].matshow(
            attribution_scores[:, i].detach().cpu().numpy(), aspect="auto", cmap="Reds"
        )
        axs[i].set_xlabel("Subnetwork Index")
        axs[i].set_ylabel("Batch Index")
        axs[i].set_title("Subnetwork Attributions")

        # Annotate each cell with the numeric value if less than 200 elements
        if attribution_scores.shape[0] * attribution_scores.shape[-1] < 200:
            for b in range(attribution_scores.shape[0]):
                for j in range(attribution_scores.shape[-1]):
                    axs[i].text(
                        j,
                        b,
                        f"{attribution_scores[b, i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                    )
    plt.colorbar(im)
    if out_dir:
        filename = (
            f"subnetwork_attributions_s{step}.png"
            if step is not None
            else "subnetwork_attributions.png"
        )
        fig.savefig(out_dir / filename, dpi=200)
    return fig


def resid_mlp_plot_results_fn(
    model: ResidualMLPSPDModel,
    target_model: ResidualMLPModel,
    step: int | None,
    out_dir: Path | None,
    device: str,
    config: Config,
    gates: dict[str, Gate],
    masks: dict[str, Float[Tensor, "batch_size m"]] | None,
    **_,
) -> dict[str, plt.Figure]:
    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    fig_dict = {}

    fig_dict["masks"], all_perm_indices = plot_mask_vals(
        model=model, target_model=target_model, gates=gates, device=device, input_magnitude=0.75
    )
    fig_dict["AB_matrices"] = plot_AB_matrices(
        model=model, device=device, all_perm_indices=all_perm_indices
    )
    return fig_dict


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    resid_mlp: ResidualMLPModel,
    resid_mlp_train_config_dict: dict[str, Any],
    label_coeffs: Float[Tensor, " n_instances"],
) -> None:
    torch.save(resid_mlp.state_dict(), out_dir / "resid_mlp.pth")

    with open(out_dir / "resid_mlp_train_config.yaml", "w") as f:
        yaml.dump(resid_mlp_train_config_dict, f, indent=2)

    with open(out_dir / "label_coeffs.json", "w") as f:
        json.dump(label_coeffs.detach().cpu().tolist(), f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "resid_mlp.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "resid_mlp_train_config.yaml"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "label_coeffs.json"), base_path=out_dir, policy="now")


def init_spd_model_from_target_model(
    model: ResidualMLPSPDModel, target_model: ResidualMLPModel, m: int
) -> None:
    """Initialize SPD model from target model.

    For mlp_in: A = target weights, B = identity
    For mlp_out: A = identity, B = target weights

    Args:
        model: The SPD model to initialize
        target_model: The target model to initialize from
        m: The number of components (must equal d_mlp for initialization)
    """
    # For ResidualMLP, we need to initialize each layer's mlp_in and mlp_out components
    for i in range(target_model.config.n_layers):
        # For mlp_in, m must equal d_mlp
        # TODO: This is broken, we shouldn't need m=d_mlp for this function.
        assert (
            m == target_model.config.d_mlp or m == target_model.config.d_embed
        ), "m must be equal to d_mlp or d_embed"

        # For mlp_in: A = target weights, B = identity
        model.layers[i].mlp_in.A.data[:] = target_model.layers[i].mlp_in.weight.data.clone()
        model.layers[i].mlp_in.B.data[:] = einops.repeat(
            torch.eye(m),
            "m d_out -> n_instances m d_out",
            n_instances=target_model.config.n_instances,
        )

        # For mlp_out: A = identity, B = target weights
        model.layers[i].mlp_out.A.data[:] = einops.repeat(
            torch.eye(m),
            "d_in m -> n_instances d_in m",
            n_instances=target_model.config.n_instances,
        )
        model.layers[i].mlp_out.B.data[:] = target_model.layers[i].mlp_out.weight.data.clone()

    logger.info("Initialized SPD model from target model")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    print(f"Using device: {device}")
    assert isinstance(config.task_config, ResidualMLPTaskConfig)

    target_model, target_model_train_config_dict, label_coeffs = ResidualMLPModel.from_pretrained(
        config.task_config.pretrained_model_path
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(
        config,
        n_features=target_model.config.n_features,
        n_layers=target_model.config.n_layers,
        d_resid=target_model.config.d_embed,
        d_mlp=target_model.config.d_mlp,
        m=config.m,
        init_scale=config.task_config.init_scale,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        resid_mlp=target_model,
        resid_mlp_train_config_dict=target_model_train_config_dict,
        label_coeffs=label_coeffs,
    )

    # Create the SPD model
    model_config = ResidualMLPSPDConfig(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
        d_embed=target_model.config.d_embed,
        d_mlp=target_model.config.d_mlp,
        n_layers=target_model.config.n_layers,
        act_fn_name=target_model.config.act_fn_name,
        apply_output_act_fn=target_model.config.apply_output_act_fn,
        in_bias=target_model.config.in_bias,
        out_bias=target_model.config.out_bias,
        init_scale=config.task_config.init_scale,
        m=config.m,
        n_gate_hidden_neurons=config.n_gate_hidden_neurons,
    )
    model = ResidualMLPSPDModel(config=model_config)

    # Use the target_model's embedding matrix and don't train it further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False
    model.W_U.data[:, :] = target_model.W_U.data.detach().clone()
    model.W_U.requires_grad = False

    # Copy the biases from the target model to the SPD model and set requires_grad to False
    for i in range(target_model.config.n_layers):
        if target_model.config.in_bias:
            model.layers[i].bias1.data[:, :] = target_model.layers[i].bias1.data.detach().clone()
            model.layers[i].bias1.requires_grad = False
        if target_model.config.out_bias:
            model.layers[i].bias2.data[:, :] = target_model.layers[i].bias2.data.detach().clone()
            model.layers[i].bias2.requires_grad = False

    if config.init_from_target_model:
        init_spd_model_from_target_model(model=model, target_model=target_model, m=config.m)

    model.to(device)
    param_names = []
    for i in range(target_model.config.n_layers):
        param_names.append(f"layers.{i}.mlp_in")
        param_names.append(f"layers.{i}.mlp_out")

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = ResidualMLPDataset(
        n_instances=model.config.n_instances,
        n_features=model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=synced_inputs,
    )

    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=out_dir,
        plot_results_fn=resid_mlp_plot_results_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

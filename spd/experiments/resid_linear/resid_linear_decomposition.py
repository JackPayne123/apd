"""Residual Linear decomposition script."""

from pathlib import Path

import fire
import torch
import wandb

from spd.experiments.resid_linear.models import ResidualLinearModel, ResidualLinearSPDFullRankModel
from spd.experiments.resid_linear.resid_linear_dataset import (
    ResidualLinearDataset,
)
from spd.log import logger
from spd.run_spd import Config, ResidualLinearConfig, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    init_wandb,
    load_config,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config, n_features: int, n_layers: int, d_resid: int, d_mlp: int) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, ResidualLinearConfig)
        run_suffix = f"seed{config.seed}_"
        if config.pnorm is not None:
            run_suffix += f"p{config.pnorm:.2e}_"
        if config.lp_sparsity_coeff is not None:
            run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
        if config.topk is not None:
            run_suffix += f"topk{config.topk:.2e}_"
        if config.topk_recon_coeff is not None:
            run_suffix += f"topkrecon{config.topk_recon_coeff:.2e}_"
        if config.topk_l2_coeff is not None:
            run_suffix += f"topkl2_{config.topk_l2_coeff:.2e}_"
        run_suffix += f"lr{config.lr:.2e}_"
        run_suffix += f"bs{config.batch_size}_"
        run_suffix += f"ft{n_features}_lay{n_layers}_resid{d_resid}_mlp{d_mlp}"
    return config.wandb_run_name_prefix + run_suffix


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        save_config_to_wandb(config)

    set_seed(config.seed)
    logger.info(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert isinstance(config.task_config, ResidualLinearConfig)

    target_model = ResidualLinearModel.from_pretrained(config.task_config.pretrained_model_path).to(
        device
    )

    run_name = get_run_name(
        config,
        n_features=target_model.n_features,
        n_layers=target_model.n_layers,
        d_resid=target_model.d_embed,
        d_mlp=target_model.d_mlp,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ResidualLinearSPDFullRankModel(
        n_features=target_model.n_features,
        d_embed=target_model.d_embed,
        d_mlp=target_model.d_mlp,
        n_layers=target_model.n_layers,
        k=config.task_config.k,
        init_scale=config.task_config.init_scale,
    ).to(device)

    # Use the target_model's embedding matrix and don't train it further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False

    param_map = {}
    for i in range(target_model.n_layers):
        # Map from pretrained model's `all_decomposable_params` to the SPD models'
        # `all_subnetwork_params_summed`.
        param_map[f"layers.{i}.input_layer.weight"] = f"layers.{i}.input_layer.weight"
        param_map[f"layers.{i}.input_layer.bias"] = f"layers.{i}.input_layer.bias"
        param_map[f"layers.{i}.output_layer.weight"] = f"layers.{i}.output_layer.weight"
        param_map[f"layers.{i}.output_layer.bias"] = f"layers.{i}.output_layer.bias"

    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        label_fn_seed=config.task_config.label_fn_seed,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        pretrained_model=target_model,
        param_map=param_map,
        out_dir=out_dir,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

from pathlib import Path

import torch

from spd.experiments.resid_linear.models import ResidualLinearModel, ResidualLinearSPDFullRankModel
from spd.experiments.resid_linear.resid_linear_dataset import ResidualLinearDataset
from spd.run_spd import Config, ResidualLinearConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, set_seed

# Create a simple ResidualLinear config that we can use in multiple tests
RESID_LINEAR_TASK_CONFIG = ResidualLinearConfig(
    task_name="residual_linear",
    init_scale=1.0,
    k=6,
    feature_probability=0.2,
    label_fn_seed=0,
    pretrained_model_path=Path(),  # We'll create this later
)


def test_resid_linear_decomposition_happy_path() -> None:
    set_seed(0)
    n_features = 3
    d_embed = 2
    d_mlp = 3
    n_layers = 1

    device = "cpu"
    config = Config(
        seed=0,
        full_rank=True,
        unit_norm_matrices=False,
        topk=1,
        batch_topk=True,
        param_match_coeff=1.0,
        topk_recon_coeff=1,
        topk_l2_coeff=1e-2,
        ablation_attributions=False,
        lr=1e-2,
        batch_size=8,
        steps=5,  # Run only a few steps for the test
        print_freq=2,
        image_freq=5,
        slow_images=False,
        save_freq=None,
        lr_warmup_pct=0.01,
        lr_schedule="cosine",
        task_config=RESID_LINEAR_TASK_CONFIG,
    )

    assert isinstance(config.task_config, ResidualLinearConfig)
    # Create a pretrained model
    pretrained_model = ResidualLinearModel(
        n_features=n_features, d_embed=d_embed, d_mlp=d_mlp, n_layers=n_layers
    ).to(device)

    # Create the SPD model
    model = ResidualLinearSPDFullRankModel(
        n_features=n_features,
        d_embed=d_embed,
        d_mlp=d_mlp,
        n_layers=n_layers,
        k=config.task_config.k,
        init_scale=config.task_config.init_scale,
    ).to(device)

    # Use the pretrained model's embedding matrix and don't train it further
    model.W_E.data[:, :] = pretrained_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False

    # Create dataset and dataloader
    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        label_fn_seed=config.task_config.label_fn_seed,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Set up param_map
    param_map = {}
    for i in range(pretrained_model.n_layers):
        param_map[f"layers.{i}.input_layer.weight"] = f"layers.{i}.input_layer.weight"
        param_map[f"layers.{i}.input_layer.bias"] = f"layers.{i}.input_layer.bias"
        param_map[f"layers.{i}.output_layer.weight"] = f"layers.{i}.output_layer.weight"
        param_map[f"layers.{i}.output_layer.bias"] = f"layers.{i}.output_layer.bias"

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out, _, _ = model(batch)
    initial_loss = torch.mean((labels - initial_out) ** 2).item()

    # Run optimize function
    optimize(
        model=model,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=pretrained_model,
        param_map=param_map,
        plot_results_fn=None,
    )

    # Calculate final loss
    final_out, _, _ = model(batch)
    final_loss = torch.mean((labels - final_out) ** 2).item()

    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Expected final loss to be lower than initial loss, but got {final_loss} >= {initial_loss}"

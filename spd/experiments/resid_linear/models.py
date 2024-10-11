import json
import os
from pathlib import Path

import einops
import torch
import wandb
import yaml
from jaxtyping import Bool, Float
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.models.base import Model, SPDFullRankModel
from spd.models.components import MLP, MLPComponentsFullRank
from spd.utils import init_param_


class ResidualLinearModel(Model):
    def __init__(self, n_features: int, d_embed: int, d_mlp: int, n_layers: int):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers

        self.W_E = nn.Parameter(torch.empty(n_features, d_embed))
        init_param_(self.W_E)
        # Make each feature have norm 1
        self.W_E.data.div_(self.W_E.data.norm(dim=1, keepdim=True))

        self.layers = nn.ModuleList(
            [MLP(d_model=d_embed, d_mlp=d_mlp, act_fn="gelu") for _ in range(n_layers)]
        )

    def forward(
        self, x: Float[Tensor, "batch n_features"]
    ) -> tuple[
        Float[Tensor, "batch d_embed"],
        dict[str, Float[Tensor, "batch d_embed"] | Float[Tensor, "batch d_mlp"]],
        dict[str, Float[Tensor, "batch d_embed"] | Float[Tensor, "batch d_mlp"]],
    ]:
        layer_pre_acts = {}
        layer_post_acts = {}
        residual = einops.einsum(
            x, self.W_E, "batch n_features, n_features d_embed -> batch d_embed"
        )
        for i, layer in enumerate(self.layers):
            out, pre_acts_i, post_acts_i = layer(residual)
            for k, v in pre_acts_i.items():
                layer_pre_acts[f"layers.{i}.{k}"] = v
            for k, v in post_acts_i.items():
                layer_post_acts[f"layers.{i}.{k}"] = v
            residual = residual + out

        return residual, layer_pre_acts, layer_post_acts

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ResidualLinearModel":
        params = torch.load(path, weights_only=True, map_location="cpu")
        with open(Path(path).parent / "config.json") as f:
            config = json.load(f)

        model = cls(
            n_features=config["n_features"],
            d_embed=config["d_embed"],
            d_mlp=config["d_mlp"],
            n_layers=config["n_layers"],
        )
        model.load_state_dict(params)
        return model

    def all_decomposable_params(
        self,
    ) -> dict[str, Float[Tensor, " d_out"] | Float[Tensor, "d_in d_out"]]:  # bias or weight
        """Dictionary of all parameters which will be decomposed with SPD."""
        params = {}
        for i, mlp in enumerate(self.layers):
            # We transpose because our SPD model uses (input, output) pairs, not (output, input)
            params[f"layers.{i}.input_layer.weight"] = mlp.input_layer.weight.T
            params[f"layers.{i}.input_layer.bias"] = mlp.input_layer.bias
            params[f"layers.{i}.output_layer.weight"] = mlp.output_layer.weight.T
            params[f"layers.{i}.output_layer.bias"] = mlp.output_layer.bias
        return params


class ResidualLinearSPDFullRankModel(SPDFullRankModel):
    def __init__(
        self, n_features: int, d_embed: int, d_mlp: int, n_layers: int, k: int, init_scale: float
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.k = k

        self.W_E = nn.Parameter(torch.empty(n_features, d_embed))

        self.layers = nn.ModuleList(
            [
                MLPComponentsFullRank(
                    d_embed=self.d_embed,
                    d_mlp=d_mlp,
                    k=k,
                    init_scale=init_scale,
                    in_bias=True,
                    out_bias=True,
                )
                for _ in range(n_layers)
            ]
        )

    def all_subnetwork_params(
        self,
    ) -> dict[str, Float[Tensor, "k d_out"] | Float[Tensor, "k d_in d_out"]]:  # bias or weight
        params = {}
        for i, mlp in enumerate(self.layers):
            params[f"layers.{i}.input_layer.weight"] = mlp.linear1.subnetwork_params
            params[f"layers.{i}.input_layer.bias"] = mlp.linear1.bias
            params[f"layers.{i}.output_layer.weight"] = mlp.linear2.subnetwork_params
            params[f"layers.{i}.output_layer.bias"] = mlp.linear2.bias
        return params

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "k d_out"] | Float[Tensor, "k d_in d_out"]]:  # bias or weight
        return {p_name: p.sum(dim=0) for p_name, p in self.all_subnetwork_params().items()}

    def forward(
        self, x: Float[Tensor, "batch n_features"], topk_mask: Bool[Tensor, "batch k"] | None = None
    ) -> tuple[
        Float[Tensor, "batch d_embed"],
        dict[str, Float[Tensor, "batch d_embed"] | Float[Tensor, "batch d_mlp"]],
        dict[str, Float[Tensor, "batch k d_embed"]],
    ]:
        """
        Returns:
            x: The output of the model
            layer_acts: A dictionary of activations for each layer in each MLP.
            inner_acts: A dictionary of component activations (just after the A matrix) for each
                layer in each MLP.
        """
        layer_acts = {}
        inner_acts = {}
        residual = einops.einsum(
            x, self.W_E, "batch n_features, n_features d_embed -> batch d_embed"
        )
        for i, layer in enumerate(self.layers):
            layer_out, layer_acts_i, inner_acts_i = layer(residual, topk_mask)
            assert len(layer_acts_i) == len(inner_acts_i) == 2
            residual = residual + layer_out
            layer_acts[f"layers.{i}.input_layer.weight"] = layer_acts_i[0]
            layer_acts[f"layers.{i}.output_layer.weight"] = layer_acts_i[1]
            inner_acts[f"layers.{i}.input_layer.weight"] = inner_acts_i[0]
            inner_acts[f"layers.{i}.output_layer.weight"] = inner_acts_i[1]
        return residual, layer_acts, inner_acts

    @classmethod
    def from_pretrained(
        cls, path: str | Path, config_file: str | Path | None = None
    ) -> "ResidualLinearSPDFullRankModel":
        """Instantiate from a checkpoint file and (optionally) a config file.

        Args:
            path: The path to the checkpoint file.
            config_file: The path to the yaml config file. If not provided, the config will be
                loaded from the checkpoint file.
        """
        path = Path(path)

        config_file_path = (
            path.parent / "final_config.yaml" if config_file is None else Path(config_file)
        )
        with open(config_file_path) as f:
            config_dict = yaml.safe_load(f)

        params = torch.load(path, weights_only=True, map_location="cpu")

        # Hackily get the config values which aren't stored in the SPD config from the params
        # TODO: In future we should store the target model config when running SPD.
        n_features, d_embed = params["W_E"].shape
        n_layers = max(int(k.split(".")[1]) for k in params if "layers." in k) + 1
        k_2, d_embed_2, d_mlp = params["layers.0.linear1.subnetwork_params"].shape
        assert k_2 == config_dict["task_config"]["k"], "k doesn't match between params and config"
        assert d_embed == d_embed_2, "d_embed does not match between W_E and the first linear layer"

        model = cls(
            n_features=n_features,
            d_embed=d_embed,
            d_mlp=d_mlp,
            n_layers=n_layers,
            k=config_dict["task_config"]["k"],
            init_scale=config_dict["task_config"]["init_scale"],
        )
        model.load_state_dict(params)
        return model

    @classmethod
    def from_wandb(cls, wandb_project_run_id: str) -> "ResidualLinearSPDFullRankModel":
        """Instantiate ResidualLinearSPDFullRankModel using the latest checkpoint from a wandb run.

        Args:
            wandb_project_run_id: The wandb project name and run ID separated by a forward slash.
                E.g. "gpt2/2lzle2f0"

        Returns:
            A ResidualLinearSPDFullRankModel instance loaded from the specified wandb run.
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        cache_dir = Path(os.environ.get("SPD_CACHE_DIR", "/tmp/"))
        model_cache_dir = cache_dir / wandb_project_run_id

        train_config_file_remote = [
            file for file in run.files() if file.name.endswith("final_config.yaml")
        ][0]

        train_config_file = train_config_file_remote.download(
            exist_ok=True, replace=True, root=model_cache_dir
        ).name

        checkpoints = [file for file in run.files() if file.name.endswith(".pth")]
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint files found in run {wandb_project_run_id}")
        latest_checkpoint_remote = sorted(
            checkpoints, key=lambda x: int(x.name.split(".pth")[0].split("_")[-1])
        )[-1]
        latest_checkpoint_file = latest_checkpoint_remote.download(
            exist_ok=True, replace=True, root=model_cache_dir
        ).name
        assert latest_checkpoint_file is not None, "Failed to download the latest checkpoint."

        return cls.from_pretrained(path=latest_checkpoint_file, config_file=train_config_file)

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, " d_out"] | Float[Tensor, "d_in d_out"]]:  # bias or weight
        stored_vals = {}
        for i, mlp in enumerate(self.layers):
            stored_vals[f"layers.{i}.input_layer.weight"] = (
                mlp.linear1.subnetwork_params[subnet_idx, :, :].detach().clone()
            )
            stored_vals[f"layers.{i}.input_layer.bias"] = (
                mlp.linear1.bias[subnet_idx, :].detach().clone()
            )
            stored_vals[f"layers.{i}.output_layer.weight"] = (
                mlp.linear2.subnetwork_params[subnet_idx, :, :].detach().clone()
            )
            stored_vals[f"layers.{i}.output_layer.bias"] = (
                mlp.linear2.bias[subnet_idx, :].detach().clone()
            )
            mlp.linear1.subnetwork_params[subnet_idx, :, :] = 0.0
            mlp.linear1.bias[subnet_idx, :] = 0.0
            mlp.linear2.subnetwork_params[subnet_idx, :, :] = 0.0
            mlp.linear2.bias[subnet_idx, :] = 0.0
        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[str, Float[Tensor, " d_out"] | Float[Tensor, "d_in d_out"]],
    ) -> None:
        for i, mlp in enumerate(self.layers):
            mlp.linear1.subnetwork_params[subnet_idx, :, :] = stored_vals[
                f"layers.{i}.input_layer.weight"
            ]
            mlp.linear1.bias[subnet_idx, :] = stored_vals[f"layers.{i}.input_layer.bias"]
            mlp.linear2.subnetwork_params[subnet_idx, :, :] = stored_vals[
                f"layers.{i}.output_layer.weight"
            ]
            mlp.linear2.bias[subnet_idx, :] = stored_vals[f"layers.{i}.output_layer.bias"]

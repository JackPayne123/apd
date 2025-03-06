from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.configs import Config, TMSTaskConfig
from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.models.components import (
    Gate,
    GateMLP,
    Linear,
    LinearComponent,
    TransposedLinear,
    TransposedLinearComponent,
)
from spd.types import WANDB_PATH_PREFIX, ModelPath
from spd.utils import replace_deprecated_param_names
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class TMSModelPaths(BaseModel):
    """Paths to output files from a TMSModel training run."""

    tms_train_config: Path
    checkpoint: Path


class TMSModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    device: str


def _tms_forward(
    x: Float[Tensor, "batch n_instances n_features"],
    linear1: Linear | LinearComponent,
    linear2: TransposedLinear | TransposedLinearComponent,
    b_final: Float[Tensor, "n_instances n_features"],
    masks: dict[str, Float[Tensor, "batch n_instances m"]] | None = None,
    hidden_layers: nn.ModuleList | None = None,
) -> Float[Tensor, "batch n_instances n_features"]:
    """Forward pass used for TMSModel and TMSSPDModel.

    Note that masks have no effect for TMSModel.
    """
    linear1_mask = masks["linear1"] if masks is not None else None
    hidden = linear1(x, mask=linear1_mask)
    if hidden_layers is not None:
        for i, layer in enumerate(hidden_layers):
            hidden_mask = masks[f"hidden_layers.{i}"] if masks is not None else None
            hidden = layer(hidden, mask=hidden_mask)
    linear2_mask = masks["linear2"] if masks is not None else None
    out_pre_relu = linear2(hidden, mask=linear2_mask) + b_final
    out = F.relu(out_pre_relu)
    return out


class TMSModel(HookedRootModule):
    def __init__(self, config: TMSModelConfig):
        super().__init__()
        self.config = config

        self.linear1 = Linear(
            d_in=config.n_features,
            d_out=config.n_hidden,
            n_instances=config.n_instances,
            init_type="xavier_normal",
        )
        # Use tied weights for the second linear layer
        self.linear2 = TransposedLinear(self.linear1.weight)

        self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features)))

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList()
            for _ in range(config.n_hidden_layers):
                layer = Linear(
                    d_in=config.n_hidden,
                    d_out=config.n_hidden,
                    n_instances=config.n_instances,
                    init_type="xavier_normal",
                )
                self.hidden_layers.append(layer)
        self.setup()

    def forward(
        self, x: Float[Tensor, "... n_instances n_features"], **_: Any
    ) -> Float[Tensor, "... n_instances n_features"]:
        return _tms_forward(
            x=x,
            linear1=self.linear1,
            linear2=self.linear2,
            b_final=self.b_final,
            hidden_layers=self.hidden_layers,
        )

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> TMSModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        tms_model_config_path = download_wandb_file(run, run_dir, "tms_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return TMSModelPaths(tms_train_config=tms_model_config_path, checkpoint=checkpoint_path)

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["TMSModel", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `resid_mlp_train_config.yaml` and `label_coeffs.json` are in the same
                directory as the checkpoint.

        Returns:
            model: The pretrained TMSModel
            tms_model_config_dict: The config dict used to train the model (we don't
                instantiate a train config due to circular import issues)
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = TMSModelPaths(
                tms_train_config=Path(path).parent / "tms_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.tms_train_config) as f:
            tms_train_config_dict = yaml.safe_load(f)

        tms_config = TMSModelConfig(**tms_train_config_dict["tms_model_config"])
        tms = cls(config=tms_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        params = replace_deprecated_param_names(params, {"W": "linear1.weight"})
        tms.load_state_dict(params)

        return tms, tms_train_config_dict


class TMSSPDPaths(BaseModel):
    """Paths to output files from a TMSSPDModel training run."""

    final_config: Path
    tms_train_config: Path
    checkpoint: Path


class TMSSPDModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    device: str
    m: PositiveInt
    n_gate_hidden_neurons: PositiveInt | None = None


class TMSSPDModel(SPDModel):
    def __init__(self, config: TMSSPDModelConfig):
        super().__init__()
        self.config = config
        self.n_instances = config.n_instances  # Required for backwards compatibility
        self.n_features = config.n_features  # Required for backwards compatibility
        self.m = config.m

        self.linear1 = LinearComponent(
            d_in=config.n_features,
            d_out=config.n_hidden,
            n_instances=config.n_instances,
            init_type="xavier_normal",
            init_scale=1.0,
            m=self.m,
        )
        self.linear2 = TransposedLinearComponent(self.linear1.A, self.linear1.B)
        self.b_final = nn.Parameter(
            torch.zeros((config.n_instances, config.n_features), device=config.device)
        )

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList(
                [
                    LinearComponent(
                        d_in=config.n_hidden,
                        d_out=config.n_hidden,
                        n_instances=config.n_instances,
                        init_type="xavier_normal",
                        init_scale=1.0,
                        m=self.m,
                    )
                    for _ in range(config.n_hidden_layers)
                ]
            )

        # Use GateMLP if n_gate_hidden_neurons is provided, otherwise use Gate
        gate_class = GateMLP if config.n_gate_hidden_neurons else Gate
        gate_kwargs = {"m": self.m, "n_instances": config.n_instances}
        if config.n_gate_hidden_neurons:
            gate_kwargs["n_gate_hidden_neurons"] = config.n_gate_hidden_neurons

        self.gates = nn.ModuleDict(
            {
                "linear1": gate_class(**gate_kwargs),
                "linear2": gate_class(**gate_kwargs),
                **{
                    f"hidden_layers-{i}": gate_class(**gate_kwargs)
                    for i in range(config.n_hidden_layers)
                },
            }
        )

        self.setup()

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        masks: dict[str, Float[Tensor, "batch n_instances m"]] | None = None,
    ) -> Float[Tensor, "batch n_instances n_features"]:
        return _tms_forward(
            x=x,
            linear1=self.linear1,
            linear2=self.linear2,
            b_final=self.b_final,
            hidden_layers=self.hidden_layers,
            masks=masks,
        )

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> TMSSPDPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="spd_model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        tms_train_config_path = download_wandb_file(run, run_dir, "tms_train_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return TMSSPDPaths(
            final_config=final_config_path,
            tms_train_config=tms_train_config_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["TMSSPDModel", Config]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, the format
                must be `wandb:entity/project/run_id`. If `api.entity` is set (e.g. via setting
                WANDB_ENTITY in .env), this can be in the form `wandb:project/run_id` and if
                form `wandb:project/run_id` and if `api.project` is set this can just be
                `wandb:run_id`. If local path, assumes that `resid_mlp_train_config.yaml` and
                `label_coeffs.json` are in the same directory as the checkpoint.
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            paths = TMSSPDPaths(
                final_config=Path(path).parent / "final_config.yaml",
                tms_train_config=Path(path).parent / "tms_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.final_config) as f:
            final_config_dict = yaml.safe_load(f)

        final_config_dict.pop("post_act_recon_coeff", None)

        spd_config = Config(**final_config_dict)

        with open(paths.tms_train_config) as f:
            tms_train_config_dict = yaml.safe_load(f)

        assert isinstance(spd_config.task_config, TMSTaskConfig)
        tms_spd_config = TMSSPDModelConfig(
            **tms_train_config_dict["tms_model_config"],
            m=spd_config.m,
            n_gate_hidden_neurons=spd_config.n_gate_hidden_neurons,
        )
        model = cls(config=tms_spd_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(params)
        return model, spd_config

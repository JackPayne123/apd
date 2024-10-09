from pathlib import Path

import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.models.base import Model, SPDFullRankModel
from spd.models.components import MLP, MLPComponentsFullRank
from spd.utils import init_param_


class ResidualLinearModel(Model):
    def __init__(self, n_features: int, d_embed: int, d_mlp: int, n_layers: int):
        super().__init__()
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
        raise NotImplementedError

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
        params = {}
        for i, mlp in enumerate(self.layers):
            params[f"layers.{i}.input_layer.weight"] = mlp.linear1.subnetwork_params.sum(dim=0)
            params[f"layers.{i}.input_layer.bias"] = mlp.linear1.bias.sum(dim=0)
            params[f"layers.{i}.output_layer.weight"] = mlp.linear2.subnetwork_params.sum(dim=0)
            params[f"layers.{i}.output_layer.bias"] = mlp.linear2.bias.sum(dim=0)
        return params

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

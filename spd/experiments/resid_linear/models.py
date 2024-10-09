from pathlib import Path

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.models.base import Model
from spd.models.components import MLP
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

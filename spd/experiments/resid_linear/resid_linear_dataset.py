from collections.abc import Callable

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


def calc_labels(
    coeffs: Float[Tensor, " d_embed"], residual: Float[Tensor, "batch d_embed"]
) -> Float[Tensor, "batch n_functions"]:
    pre_act_fn = einops.einsum(coeffs, residual, "d_embed, batch d_embed -> batch d_embed")
    labels = F.gelu(pre_act_fn) + residual
    return labels


class ResidualLinearDataset(
    Dataset[tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]]
):
    def __init__(
        self,
        embed_matrix: Float[Tensor, "n_features d_embed"],
        n_features: int,
        feature_probability: float,
        device: str,
        label_fn_seed: int,
    ):
        self.embed_matrix = embed_matrix
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.label_fn_seed = label_fn_seed
        self.label_fn = self.create_label_function()

    def __len__(self) -> int:
        return 2**31

    def create_label_function(
        self,
    ) -> Callable[[Float[Tensor, " d_embed"]], Float[Tensor, " d_embed"]]:
        """Create a function that takes in the embedded inputs and returns the labels"""
        gen = torch.Generator()
        gen.manual_seed(self.label_fn_seed)
        coeffs = torch.rand(self.embed_matrix.shape[1], generator=gen)
        return lambda residual: calc_labels(coeffs, residual)

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]:
        # batch values are between [-1, 1]
        batch = torch.rand(batch_size, self.n_features, device=self.device) * 2 - 1
        mask = torch.rand_like(batch) < self.feature_probability
        batch = batch * mask
        embed = einops.einsum(
            batch, self.embed_matrix, "batch n_features, n_features d_embed -> batch d_embed"
        )
        labels = self.label_fn(embed)
        return batch, labels

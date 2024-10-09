from collections.abc import Callable

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class ResidualLinearDataset(Dataset[Float[Tensor, " n_features"]]):
    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
    ):
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, " n_features"], Float[Tensor, " n_features"]]:
        # Output values are between [-1, 1]
        batch = torch.rand(batch_size, self.n_features, device=self.device) * 2 - 1
        mask = torch.rand_like(batch) < self.feature_probability
        batch = batch * mask
        return batch, batch.clone().detach()


def calc_labels(
    coeffs: Float[Tensor, " d_resid"], residual: Float[Tensor, "batch d_resid"]
) -> Float[Tensor, "batch n_functions"]:
    pre_act_fn = einops.einsum(coeffs, residual, "d_resid, batch d_resid -> batch d_resid")
    labels = F.gelu(pre_act_fn) + residual
    return labels


def create_label_function(
    d_resid: int,
    seed: int,
) -> Callable[[Float[Tensor, " d_resid"]], Float[Tensor, " d_resid"]]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    coeffs = torch.rand(d_resid, generator=gen)
    return lambda residual: calc_labels(coeffs, residual)

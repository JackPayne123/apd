import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


def calc_labels(
    coeffs: Float[Tensor, " n_functions"],
    embed_matrix: Float[Tensor, "n_features d_embed"],
    inputs: Float[Tensor, "batch n_functions"],
) -> Float[Tensor, "batch d_embed"]:
    """Calculate the corresponding labels for the inputs using W_E(gelu(coeffs*x) + x)."""
    weighted_inputs = einops.einsum(
        inputs, coeffs, "batch n_functions, n_functions -> batch n_functions"
    )
    raw_labels = F.gelu(weighted_inputs) + inputs
    embedded_labels = einops.einsum(
        raw_labels, embed_matrix, "batch n_functions, n_functions d_embed -> batch d_embed"
    )
    return embedded_labels


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

        # Create random coeffs between [1, 2]
        gen = torch.Generator().manual_seed(self.label_fn_seed)
        self.coeffs = torch.rand(self.embed_matrix.shape[0], generator=gen) + 1

        self.label_fn = lambda inputs: calc_labels(self.coeffs, self.embed_matrix, inputs)

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]:
        # batch values are between [-1, 1]
        batch = torch.rand(batch_size, self.n_features, device=self.device) * 2 - 1
        mask = torch.rand_like(batch) < self.feature_probability
        batch = batch * mask
        labels = self.label_fn(batch)
        return batch, labels

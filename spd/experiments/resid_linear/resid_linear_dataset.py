from typing import Literal

import einops
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


def calc_labels(
    coeffs: Float[Tensor, " n_functions"],
    embed_matrix: Float[Tensor, "n_features d_embed"],
    inputs: Float[Tensor, "batch n_functions"],
    act_fn_name: Literal["gelu", "relu"] = "gelu",
) -> Float[Tensor, "batch d_embed"]:
    """Calculate the corresponding labels for the inputs using W_E(gelu(coeffs*x) + x)."""
    weighted_inputs = einops.einsum(
        inputs, coeffs, "batch n_functions, n_functions -> batch n_functions"
    )
    assert act_fn_name in ["gelu", "relu"]
    act_fn = F.gelu if act_fn_name == "gelu" else F.relu
    raw_labels = act_fn(weighted_inputs) + inputs
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
        label_fn_seed: int | None = None,
        label_coeffs: list[float] | None = None,
        data_generation_type: Literal[
            "exactly_one_active",
            "exactly_two_active",
            "exactly_three_active",
            "at_least_zero_active",
        ] = "at_least_zero_active",
        act_fn_name: Literal["gelu", "relu"] = "gelu",
    ):
        assert label_coeffs is not None or label_fn_seed is not None
        self.embed_matrix = embed_matrix.to(device)
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.label_fn_seed = label_fn_seed
        self.data_generation_type = data_generation_type

        if label_coeffs is None:
            # Create random coeffs between [1, 2]
            gen = torch.Generator(device=self.device)
            if self.label_fn_seed is not None:
                gen.manual_seed(self.label_fn_seed)
            self.coeffs = (
                torch.rand(self.embed_matrix.shape[0], generator=gen, device=self.device) + 1
            )
        else:
            self.coeffs = torch.tensor(label_coeffs, device=self.device)

        self.label_fn = lambda inputs: calc_labels(
            self.coeffs, self.embed_matrix, inputs, act_fn_name=act_fn_name
        )

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]:
        if self.data_generation_type == "exactly_one_active":
            batch = self._generate_n_feature_active_batch(batch_size, 1)
        elif self.data_generation_type == "exactly_two_active":
            batch = self._generate_n_feature_active_batch(batch_size, 2)
        elif self.data_generation_type == "exactly_three_active":
            batch = self._generate_n_feature_active_batch(batch_size, 3)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._generate_multi_feature_batch(batch_size)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        labels = self.label_fn(batch)
        return batch, labels

    def _generate_n_feature_active_batch(
        self, batch_size: int, n_active_features: int
    ) -> Float[Tensor, "batch n_features"]:
        batch = torch.zeros(batch_size, self.n_features, device=self.device)
        for i in range(batch_size):
            # Choose feature indices
            active_features = np.random.choice(self.n_features, n_active_features, replace=False)
            # Generate random values in [-1, 1] for active features
            batch[i, active_features] = torch.rand(n_active_features, device=self.device) * 2 - 1
        return batch

    def _generate_multi_feature_batch(self, batch_size: int) -> Float[Tensor, "batch n_features"]:
        # Generate random values in [-1, 1] for all features
        batch = torch.rand((batch_size, self.n_features), device=self.device) * 2 - 1
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

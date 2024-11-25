from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class ResidualMLPDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances n_features"],
        ]
    ]
):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: Literal[
            "exactly_one_active", "at_least_zero_active"
        ] = "at_least_zero_active",
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances n_features"]
    ]:
        if self.data_generation_type == "exactly_one_active":
            batch = self._generate_one_feature_active_batch(batch_size)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._generate_multi_feature_batch(batch_size)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        return batch, batch.clone().detach()

    def _generate_one_feature_active_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)

        active_features = torch.randint(
            0, self.n_features, (batch_size, self.n_instances), device=self.device
        )
        random_values = torch.rand(batch_size, self.n_instances, 1, device=self.device) * 2 - 1
        batch.scatter_(dim=2, index=active_features.unsqueeze(-1), src=random_values)
        return batch

    def _generate_multi_feature_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        # Generate random values in [-1, 1] for all features
        batch = (
            torch.rand((batch_size, self.n_instances, self.n_features), device=self.device) * 2 - 1
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

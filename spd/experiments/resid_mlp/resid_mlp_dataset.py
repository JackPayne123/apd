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
            "exactly_one_active", "exactly_two_active", "at_least_zero_active"
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
        elif self.data_generation_type == "exactly_two_active":
            batch = self._generate_two_feature_active_batch(batch_size)
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

    def _generate_two_feature_active_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)

        # Create indices for all features
        feature_indices = torch.arange(self.n_features, device=self.device)
        # Expand to batch size and n_instances
        feature_indices = feature_indices.expand(batch_size, self.n_instances, self.n_features)

        # For each instance in the batch, randomly permute the features
        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)

        # Take first two indices for each instance - guaranteed no duplicates
        active_features = permuted_features[..., :2]

        # Generate random values in [-1,1] for the active features
        random_values = torch.rand(batch_size, self.n_instances, 2, device=self.device) * 2 - 1

        # Place the first active feature
        batch.scatter_(dim=2, index=active_features[..., 0:1], src=random_values[..., 0:1])
        # Place the second active feature
        batch.scatter_(dim=2, index=active_features[..., 1:2], src=random_values[..., 1:2])

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

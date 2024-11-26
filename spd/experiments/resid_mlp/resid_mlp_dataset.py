from collections.abc import Callable
from typing import Literal

import einops
import torch
import torch.nn.functional as F
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
        calc_labels: bool = True,  # If False, just return the inputs as labels
        label_type: Literal["act_plus_resid", "abs"] | None = None,
        act_fn_name: Literal["relu", "gelu"] | None = None,
        label_fn_seed: int | None = None,
        label_coeffs: Float[Tensor, "n_instances n_features"] | None = None,
        data_generation_type: Literal[
            "exactly_one_active", "exactly_two_active", "at_least_zero_active"
        ] = "at_least_zero_active",
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.calc_labels = calc_labels
        self.label_type = label_type
        self.act_fn_name = act_fn_name
        self.label_fn_seed = label_fn_seed
        self.label_coeffs = label_coeffs
        self.data_generation_type = data_generation_type
        self.label_fn = None

        if calc_labels:
            self.label_coeffs = (
                self.calc_label_coeffs() if label_coeffs is None else label_coeffs
            ).to(self.device)

            assert label_type is not None, "Must provide label_type if calc_labels is True"
            self.label_fn = self.create_label_fn(label_type, act_fn_name)

    def __len__(self) -> int:
        return 2**31

    def create_label_fn(
        self,
        label_type: Literal["act_plus_resid", "abs"],
        act_fn_name: Literal["relu", "gelu"] | None = None,
    ) -> Callable[
        [Float[Tensor, "batch n_instances n_features"]],
        Float[Tensor, "batch n_instances n_features"],
    ]:
        if label_type == "act_plus_resid":
            assert act_fn_name in ["relu", "gelu"], "act_fn_name must be 'relu' or 'gelu'"
            return lambda batch: self.calc_act_plus_resid_labels(
                batch=batch, act_fn_name=act_fn_name
            )
        elif label_type == "abs":
            return lambda batch: self.calc_abs_labels(batch)

    def _calc_weighted_inputs(
        self,
        batch: Float[Tensor, "batch n_instances n_functions"],
        coeffs: Float[Tensor, "n_instances n_functions"],
    ) -> Float[Tensor, "batch n_instances n_functions"]:
        return einops.einsum(
            batch,
            coeffs,
            "batch n_instances n_functions, n_instances n_functions -> batch n_instances n_functions",
        )

    def calc_act_plus_resid_labels(
        self,
        batch: Float[Tensor, "batch n_instances n_functions"],
        act_fn_name: Literal["relu", "gelu"],
    ) -> Float[Tensor, "batch n_instances n_functions"]:
        """Calculate the corresponding labels for the batch using `act_fn(coeffs*x) + x`."""
        assert self.label_coeffs is not None
        weighted_inputs = self._calc_weighted_inputs(batch, self.label_coeffs)
        assert act_fn_name in ["relu", "gelu"], "act_fn_name must be 'relu' or 'gelu'"
        act_fn = F.relu if act_fn_name == "relu" else F.gelu
        labels = act_fn(weighted_inputs) + batch
        return labels

    def calc_abs_labels(
        self, batch: Float[Tensor, "batch n_instances n_features"]
    ) -> Float[Tensor, "batch n_instances n_features"]:
        assert self.label_coeffs is not None
        weighted_inputs = self._calc_weighted_inputs(batch, self.label_coeffs)
        return torch.abs(weighted_inputs)

    def calc_label_coeffs(self) -> Float[Tensor, "n_instances n_features"]:
        """Create random coeffs between [1, 2] using label_fn_seed if provided."""
        gen = torch.Generator(device=self.device)
        if self.label_fn_seed is not None:
            gen.manual_seed(self.label_fn_seed)
        return torch.rand(self.n_instances, self.n_features, generator=gen, device=self.device) + 1

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

        labels = self.label_fn(batch) if self.label_fn is not None else batch.clone().detach()
        return batch, labels

    def _generate_one_feature_active_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Batch with exactly one feature active in each sample."""
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
        """Batch with exactly two features active in each sample."""
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
        """Batch with each feature activating independently with prob `feature_probability`.

        The value of each active feature is in [-1, 1].
        """
        batch = (
            torch.rand((batch_size, self.n_instances, self.n_features), device=self.device) * 2 - 1
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

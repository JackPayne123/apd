from collections.abc import Callable
from typing import Literal

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.utils import BaseSPDDataset


class ResidualMLPDataset(BaseSPDDataset):
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
        super().__init__(
            n_instances=n_instances,
            n_features=n_features,
            feature_probability=feature_probability,
            device=device,
            data_generation_type=data_generation_type,
            value_range=(-1.0, 1.0),
        )

        self.calc_labels = calc_labels
        self.label_type = label_type
        self.act_fn_name = act_fn_name
        self.label_fn_seed = label_fn_seed
        self.label_coeffs = label_coeffs
        self.label_fn = None

        if calc_labels:
            self.label_coeffs = (
                self.calc_label_coeffs() if label_coeffs is None else label_coeffs
            ).to(self.device)

            assert label_type is not None, "Must provide label_type if calc_labels is True"
            self.label_fn = self.create_label_fn(label_type, act_fn_name)

    def generate_batch(
        self, batch_size: int
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances n_features"]
    ]:
        # Note that the parent_labels are just the batch itself
        batch, parent_labels = super().generate_batch(batch_size)
        labels = self.label_fn(batch) if self.label_fn is not None else parent_labels
        return batch, labels

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

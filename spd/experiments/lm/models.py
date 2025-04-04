"""
Defines a SSModel class that is a wrapper around a llama model from SimpleStories
"""

import fnmatch
from typing import Any

import torch.nn as nn
from jaxtyping import Float
from simple_stories_train.models.llama import Llama
from torch import Tensor

from spd.models.components import LinearComponent


class LinearComponentWithBias(nn.Module):
    """A LinearComponent with a bias parameter."""

    def __init__(self, linear_component: LinearComponent, bias: Tensor | None):
        super().__init__()
        self.linear_component = linear_component
        self.bias = bias
        self.mask: Float[Tensor, "... m"] | None = None  # Gets set on sparse forward passes

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        # Note: We assume bias is added *after* the component multiplication
        # Also assume input is (batch, seq_len, d_in)
        out = self.linear_component(x, mask=self.mask)
        if self.bias is not None:
            out += self.bias
        return out


def nn_linear_to_components(linear_module: nn.Linear, m: int) -> LinearComponentWithBias:
    """Replace a nn.Linear module with a LinearComponentWithBias module."""
    d_out, d_in = linear_module.weight.shape

    linear_component = LinearComponent(d_in=d_in, d_out=d_out, m=m, n_instances=None)

    # # Initialize with A = W (original weights) and B = I (identity)
    # # This provides a starting point where the component exactly equals the original
    # linear_component.A.data[:] = linear_module.weight.t()  # (d_in, m)
    # linear_component.B.data[:] = torch.eye(m)

    bias = linear_module.bias.clone() if linear_module.bias is not None else None  # type: ignore

    return LinearComponentWithBias(linear_component, bias)


# class SSModel(HookedRootModule):
class SSModel(nn.Module):
    """Wrapper around a llama model from SimpleStories for running SPD."""

    def __init__(self, llama_model: Llama, target_module_patterns: list[str], m: int):
        super().__init__()
        self.model = llama_model
        self.components = self.create_target_components(
            target_module_patterns=target_module_patterns, m=m
        )
        # self.setup()

    def create_target_components(
        self, target_module_patterns: list[str], m: int
    ) -> dict[str, LinearComponentWithBias]:
        """Create target components for the model."""
        components = {}
        for name, module in self.model.named_modules():
            for pattern in target_module_patterns:
                if fnmatch.fnmatch(name, pattern):
                    assert isinstance(module, nn.Linear), (
                        f"Module '{name}' matched pattern '{pattern}' but is not nn.Linear. "
                        f"Found type: {type(module)}"
                    )
                    components[name] = nn_linear_to_components(module, m=m)
                    break
        return components

    def to(self, *args: Any, **kwargs: Any) -> "SSModel":
        """Move the model and components to a device."""
        self.model.to(*args, **kwargs)
        for component in self.components.values():
            component.to(*args, **kwargs)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Regular forward pass of the (target) model."""
        return self.model(*args, **kwargs)

    def forward_with_components(
        self,
        *args: Any,
        masks: dict[str, Float[Tensor, "batch pos m"]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with temporary component replacement."""
        old_modules = {}
        for module_name, component in self.components.items():
            old_module = self.model.get_submodule(module_name)
            assert old_module is not None
            old_modules[module_name] = old_module

            if masks is not None:
                assert module_name in masks, f"Mask for {module_name} not found"
                component.mask = masks[module_name]
            self.model.set_submodule(module_name, component)

        out = self.model(*args, **kwargs)

        # Restore the original modules
        for module_name, old_module in old_modules.items():
            self.model.set_submodule(module_name, old_module)

        # Remove the masks attribute from the components
        for component in self.components.values():
            component.mask = None

        return out

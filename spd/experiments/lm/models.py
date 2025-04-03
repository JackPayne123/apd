"""
Defines a SSModel class that is a wrapper around a llama model from SimpleStories
"""

from typing import Any

import torch.nn as nn
from jaxtyping import Float
from simple_stories_train.models.llama import Llama
from torch import Tensor

from spd.models.components import LinearComponent
from spd.module_utils import get_nested_module_attr, set_nested_module_attr


class LinearComponentWithBias(nn.Module):
    """A LinearComponent with a bias parameter."""

    def __init__(self, linear_component: LinearComponent, bias: Tensor | None):
        super().__init__()
        self.linear_component = linear_component
        self.bias = bias
        self.mask: Float[Tensor, "batch pos m"] | None = None  # Gets set on sparse forward passes

    def forward(self, x: Float[Tensor, "batch d_in"]) -> Float[Tensor, "batch d_out"]:
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


# Create LinearComponentWithBias objects for gate_proj in each layer
def create_gate_proj_components(model: Llama, rank: int) -> dict[str, LinearComponentWithBias]:
    components = {}
    for i in range(len(model.transformer.h)):
        gate_proj = model.transformer.h[i].mlp.gate_proj
        module_name = f"model.transformer.h.{i}.mlp.gate_proj"
        components[module_name] = nn_linear_to_components(gate_proj, m=rank)
    return components


class SSModel(nn.Module):
    """Wrapper around a llama model from SimpleStories for running SPD."""

    def __init__(self, llama_model: Llama):
        super().__init__()
        self.model = llama_model

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Regular forward pass of the (target) model."""
        return self.model(*args, **kwargs)

    def forward_with_components(
        self,
        *args: Any,
        components: dict[str, LinearComponentWithBias],
        masks: dict[str, Float[Tensor, "batch pos m"]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with temporary component replacement."""
        old_modules = {}
        for module_name, component in components.items():
            old_module = get_nested_module_attr(self, module_name)
            assert old_module is not None
            old_modules[module_name] = old_module

            if masks is not None:
                assert module_name in masks, f"Mask for {module_name} not found"
                component.mask = masks[module_name]
            set_nested_module_attr(self, module_name, component)

        out = self.model(*args, **kwargs)

        # Restore the original modules
        for module_name, old_module in old_modules.items():
            set_nested_module_attr(self, module_name, old_module)

        # Remove the masks attribute from the components
        for component in components.values():
            component.mask = None

        return out

"""Calculations for how important each component is to the output."""

import einops
import torch
from jaxtyping import Float
from torch import Tensor


def calc_grad_attributions(
    target_out: Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"],
    pre_weight_acts: dict[
        str, Float[Tensor, "batch d_in"] | Float[Tensor, "batch n_instances d_in"]
    ],
    post_weight_acts: dict[
        str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]
    ],
    Bs: dict[str, Float[Tensor, "m d_out"] | Float[Tensor, "n_instances m d_out"]],
    target_component_acts: dict[
        str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]
    ],
) -> dict[str, Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the product of the gradient of the target model output w.r.t. the post acts
    and the component_acts. I.e.
        sum_i[((pre_weight_acts @ A) * (B @ d(out_i)/d(post_weight_acts))) ** 2]

    Note: This code may be run in between the training forward pass, and the loss.backward() and
    opt.step() calls; it must not mess with the training. The reason the current implementation is
    fine to run anywhere is that we just use autograd rather than backward which does not
    populate the .grad attributes.

    Unrelatedly, we use retain_graph=True in a bunch of cases where we want to later use the `out`
    variable in e.g. the loss function.

    Args:
        target_out: The output of the target model.
        pre_weight_acts: The activations of the target model before the weight matrix at each layer.
        post_weight_acts: The activations at the target model after the weight matrix at each layer.
        Bs: The B matrix at each layer.
        target_component_acts: The component acts at each layer. (I.e. (pre_weight_acts @ A))

    Returns:
        A dictionary of the sum of the (squared) attributions from each output dimension for each
        layer.
    """
    # Ensure that all keys are the same after removing the hook suffixes
    post_weight_act_names = [comp.removesuffix(".hook_post") for comp in post_weight_acts]
    pre_weight_act_names = [comp.removesuffix(".hook_pre") for comp in pre_weight_acts]
    assert (
        set(post_weight_act_names)
        == set(pre_weight_act_names)
        == set(Bs.keys())
        == set(target_component_acts.keys())
    )

    m = next(iter(Bs.values())).shape[-2]
    attr_shape = target_out.shape[:-1] + (m,)  # (batch, m) or (batch, n_instances, m)
    attributions: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]] = {
        param_name: torch.zeros(attr_shape, device=target_out.device, dtype=target_out.dtype)
        for param_name in post_weight_act_names
    }

    for feature_idx in range(target_out.shape[-1]):  # Iterate over the output dimensions
        grad_post_weight_acts: tuple[
            Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"], ...
        ] = torch.autograd.grad(
            target_out[..., feature_idx].sum(), list(post_weight_acts.values()), retain_graph=True
        )
        for i, param_name in enumerate(post_weight_act_names):
            # (B @ d(out)/d(post_weight_acts))
            grad_B = einops.einsum(
                Bs[param_name], grad_post_weight_acts[i], "... m d_out, ... d_out -> ... m"
            )
            attributions[param_name] += (target_component_acts[param_name] * grad_B) ** 2

    # Take the square root of each attribution and divide by the number of output dimensions
    for param_name in attributions:
        attributions[param_name] = attributions[param_name].sqrt() / target_out.shape[-1]

    return attributions

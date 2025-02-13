"""Calculations for how important each component is to the output."""

import einops
import torch
from jaxtyping import Float
from torch import Tensor

from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.module_utils import collect_nested_module_attrs


def calc_grad_attributions(
    target_out: Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"],
    pre_weight_acts: dict[
        str, Float[Tensor, "batch d_in"] | Float[Tensor, "batch n_instances d_in"]
    ],
    post_weight_acts: dict[
        str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]
    ],
    component_acts: dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]],
    Bs: dict[str, Float[Tensor, "m d_out"] | Float[Tensor, "n_instances m d_out"]],
) -> dict[str, Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the product of the gradient of the target model output w.r.t. the post acts
    and the component_acts. I.e.
        sum_i[((pre_weight_acts @ A) @ (B @ d(out_i)/d(post_weight_acts))) ** 2]

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
        component_acts: The activations after multiplying by A at each layer.
        Bs: The B matrix at each layer.

    Returns:
        A dictionary of the sum of the (squared) attributions from each output dimension for each
        layer.
    """
    # Ensure that all keys are the same after removing the hook suffixes
    post_weight_act_names = [comp.removesuffix(".hook_post") for comp in post_weight_acts]
    pre_weight_act_names = [comp.removesuffix(".hook_pre") for comp in pre_weight_acts]
    component_act_names = [comp.removesuffix(".hook_component_acts") for comp in component_acts]
    assert set(post_weight_act_names) == set(pre_weight_act_names) == set(component_act_names)

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
            attributions[param_name] += (
                component_acts[param_name + ".hook_component_acts"] * grad_B
            ) ** 2

    return attributions


def collect_subnetwork_attributions(
    spd_model: SPDModel,
    target_model: HookedRootModule,
    device: str,
    n_instances: int | None = None,
) -> dict[str, Float[Tensor, "batch m"] | Float[Tensor, "batch n_instances m"]]:
    """
    Collect subnetwork attributions.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the attributions.

    Args:
        spd_model: The model to collect attributions on.
        config: The main SPD config.
        target_model: The target model to collect attributions on.
        device: The device to run computations on.
        n_instances: The number of instances in the batch.

    Returns:
        The attribution scores.
    """
    test_batch = torch.eye(spd_model.n_features, device=device)
    if n_instances is not None:
        test_batch = einops.repeat(
            test_batch, "batch n_features -> batch n_instances n_features", n_instances=n_instances
        )
    target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
    target_out, target_cache = target_model.run_with_cache(
        test_batch, names_filter=target_cache_filter
    )

    attribution_scores = calc_grad_attributions(
        target_out=target_out,
        pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
        post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
        component_acts={k: v for k, v in target_cache.items() if k.endswith("hook_component_acts")},
        Bs=collect_nested_module_attrs(spd_model, attr_name="B", include_attr_name=False),
    )
    return attribution_scores

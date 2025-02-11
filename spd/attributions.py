"""Calculating and collecting attributions"""

from typing import Literal

import einops
import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config
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
    component_weights: dict[
        str, Float[Tensor, "C d_in d_out"] | Float[Tensor, "n_instances C d_in d_out"]
    ],
    C: int,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the product of the gradient of the target model output w.r.t. the post acts
    and the inner acts (i.e. the output of each subnetwork before being summed).

    Note that we don't use the component_acts collected from the SPD model, because this includes the
    computational graph of the full model. We only want the subnetwork parameters of the current
    layer to be in the computational graph. To do this, we multiply a detached version of the
    pre_weight_acts by the subnet parameters.

    Note: This code may be run in between the training forward pass, and the loss.backward() and
    opt.step() calls; it must not mess with the training. The reason the current implementation is
    fine to run anywhere is that we just use autograd rather than backward which does not
    populate the .grad attributes. Unrelatedly, we use retain_graph=True in a bunch of cases
    where we want to later use the `out` variable in e.g. the loss function.

    Args:
        target_out: The output of the target model.
        pre_weight_acts: The activations at the output of each subnetwork before being summed.
        post_weight_acts: The activations at the output of each layer after being summed.
        component_weights: The component weight matrix at each layer.
        C: The number of components.
    Returns:
        The sum of the (squared) attributions from each output dimension.
    """
    # Ensure that all keys are the same after removing the hook suffixes
    post_weight_act_names = [C.removesuffix(".hook_post") for C in post_weight_acts]
    pre_weight_act_names = [C.removesuffix(".hook_pre") for C in pre_weight_acts]
    component_weight_names = list(component_weights.keys())
    assert set(post_weight_act_names) == set(pre_weight_act_names) == set(component_weight_names)

    attr_shape = target_out.shape[:-1] + (C,)  # (batch, C) or (batch, n_instances, C)
    attribution_scores: Float[Tensor, "batch ... C"] = torch.zeros(
        attr_shape, device=target_out.device, dtype=target_out.dtype
    )

    component_acts = {}
    for param_name in pre_weight_act_names:
        component_acts[param_name] = einops.einsum(
            pre_weight_acts[param_name + ".hook_pre"].detach().clone(),
            component_weights[param_name],
            "... d_in, ... C d_in d_out -> ... C d_out",
        )
    out_dim = target_out.shape[-1]
    for feature_idx in range(out_dim):
        feature_attributions: Float[Tensor, "batch ... C"] = torch.zeros(
            attr_shape, device=target_out.device, dtype=target_out.dtype
        )
        grad_post_weight_acts: tuple[Float[Tensor, "batch ... d_out"], ...] = torch.autograd.grad(
            target_out[..., feature_idx].sum(), list(post_weight_acts.values()), retain_graph=True
        )
        for i, param_name in enumerate(post_weight_act_names):
            feature_attributions += einops.einsum(
                grad_post_weight_acts[i],
                component_acts[param_name],
                "... d_out ,... C d_out -> ... C",
            )

        attribution_scores += feature_attributions**2

    return attribution_scores


def collect_subnetwork_attributions(
    spd_model: SPDModel,
    config: Config,
    target_model: HookedRootModule,
    device: str,
    n_instances: int | None = None,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
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

    attribution_scores = calculate_attributions(
        model=spd_model,
        config=config,
        batch=test_batch,
        target_out=target_out,
        pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
        post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
    )
    return attribution_scores


@torch.inference_mode()
def calc_ablation_attributions(
    spd_model: SPDModel,
    batch: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    out: Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"] | None,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the attributions by ablating each subnetwork one at a time."""
    assert out is not None, "out tensor is missing."
    attr_shape = out.shape[:-1] + (spd_model.C,)  # (batch, C) or (batch, n_instances, C)
    has_instance_dim = len(out.shape) == 3
    attributions = torch.zeros(attr_shape, device=out.device, dtype=out.dtype)
    for subnet_idx in range(spd_model.C):
        stored_vals = spd_model.set_subnet_to_zero(subnet_idx, has_instance_dim)
        ablation_out, _, _ = spd_model(batch)
        out_recon = ((out - ablation_out) ** 2).mean(dim=-1)
        attributions[..., subnet_idx] = out_recon
        spd_model.restore_subnet(subnet_idx, stored_vals, has_instance_dim)
    return attributions


def calc_activation_attributions(
    component_acts: dict[
        str, Float[Tensor, "batch C d_out"] | Float[Tensor, "batch n_instances C d_out"]
    ]
    | None,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the attributions by taking the L2 norm of the activations in each subnetwork.

    Args:
        component_acts: The activations at the output of each subnetwork before being summed.
    Returns:
        The attributions for each subnetwork.
    """
    assert component_acts is not None, "Component_acts are missing"
    first_param = component_acts[next(iter(component_acts.keys()))]
    assert len(first_param.shape) in (3, 4)

    attribution_scores: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"] = (
        torch.zeros(first_param.shape[:-1], device=first_param.device, dtype=first_param.dtype)
    )
    for param_matrix in component_acts.values():
        attribution_scores += param_matrix.pow(2).sum(dim=-1)
    return attribution_scores


def calculate_attributions(
    model: SPDModel,
    config: Config,
    batch: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    target_out: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    pre_weight_acts: dict[
        str, Float[Tensor, "batch d_in"] | Float[Tensor, "batch n_instances d_in"]
    ],
    post_weight_acts: dict[
        str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]
    ],
    component_acts: dict[str, Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]]
    | None = None,
    out: Float[Tensor, "batch n_features"]
    | Float[Tensor, "batch n_instances n_features"]
    | None = None,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    attributions = None
    attribution_type: Literal["ablation", "gradient", "activation"] = config.attribution_type
    if attribution_type == "ablation":
        attributions = calc_ablation_attributions(spd_model=model, batch=batch, out=out)
    elif attribution_type == "gradient":
        component_weights = collect_nested_module_attrs(
            model, attr_name="component_weights", include_attr_name=False
        )
        attributions = calc_grad_attributions(
            target_out=target_out,
            pre_weight_acts=pre_weight_acts,
            post_weight_acts=post_weight_acts,
            component_weights=component_weights,
            C=model.C,
        )
    elif attribution_type == "activation":
        attributions = calc_activation_attributions(component_acts=component_acts)
    else:
        raise ValueError(f"Invalid attribution type: {attribution_type}")
    return attributions

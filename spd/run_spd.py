"""Run SPD on a model."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, Self

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from jaxtyping import Bool, Float
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from spd.log import logger
from spd.models.base import Model, SPDFullRankModel, SPDModel
from spd.types import Probability, RootPath
from spd.utils import (
    calc_ablation_attributions,
    calc_attributions_full_rank,
    calc_attributions_rank_one,
    calc_topk_mask,
)

# torch.set_float32_matmul_precision("high")


class TMSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = "tms"
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_instances: PositiveInt
    k: PositiveInt
    feature_probability: Probability
    train_bias: bool
    bias_val: float
    pretrained_model_path: RootPath | None = None


class DeepLinearConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["deep_linear"] = "deep_linear"
    n_features: PositiveInt | None = None
    n_layers: PositiveInt | None = None
    n_instances: PositiveInt | None = None
    k: PositiveInt | None = None
    pretrained_model_path: RootPath | None = None


class PiecewiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["piecewise"] = "piecewise"
    n_functions: PositiveInt
    neurons_per_function: PositiveInt
    n_layers: PositiveInt
    feature_probability: Probability
    range_min: float
    range_max: float
    k: PositiveInt
    target_seed: int | None = None
    dataset_seed: int | None = None
    simple_bias: bool = False
    handcoded_AB: bool = False


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    full_rank: bool = False
    seed: int = 0
    topk: PositiveFloat | None = None
    batch_topk: bool = True
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    image_freq: PositiveInt | None = None
    slow_images: bool = False
    save_freq: PositiveInt | None = None
    lr: PositiveFloat
    orthog_coeff: NonNegativeFloat | None = None
    out_recon_coeff: NonNegativeFloat | None = None
    param_match_coeff: NonNegativeFloat | None = 1.0
    topk_recon_coeff: NonNegativeFloat | None = None
    topk_l2_coeff: NonNegativeFloat | None = None
    lp_sparsity_coeff: NonNegativeFloat | None = None
    pnorm: PositiveFloat | None = None
    pnorm_end: PositiveFloat | None = None
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    lr_exponential_halflife: PositiveFloat | None = None
    lr_warmup_pct: Probability = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    sparsity_warmup_pct: Probability = 0.0
    unit_norm_matrices: bool = True
    ablation_attributions: bool = False
    task_config: DeepLinearConfig | PiecewiseConfig | TMSConfig = Field(
        ..., discriminator="task_name"
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Check valid combinations of topk and batch_size
        if self.topk is not None:
            if self.batch_topk:
                if not (self.batch_size * self.topk).is_integer():
                    logger.warning(
                        f"batch_size * topk={self.batch_size * self.topk} is not an integer, will "
                        f"round down from {self.batch_size * self.topk} to "
                        f"{int(self.batch_size * self.topk)} when calculating topk_mask"
                    )
            else:
                if not self.topk.is_integer():
                    raise ValueError("topk must be an integer when not using batch_topk")

        # Warn if neither topk_recon_coeff nor lp_sparsity_coeff is set
        if not self.topk_recon_coeff and not self.lp_sparsity_coeff:
            logger.warning("Neither topk_recon_coeff nor lp_sparsity_coeff is set")

        # If topk_recon_coeff is set, topk must be set
        if self.topk_recon_coeff is not None:
            assert self.topk is not None, "topk must be set if topk_recon_coeff is set"

        # If lp_sparsity_coeff is set, pnorm or pnorm_end must be set
        if self.lp_sparsity_coeff is not None:
            assert (
                self.pnorm is not None or self.pnorm_end is not None
            ), "pnorm or pnorm_end must be set if lp_sparsity_coeff is set"

        # Check that topk_l2_coeff and topk_recon_coeff are None if topk is None
        if self.topk is None:
            assert self.topk_l2_coeff is None, "topk_l2_coeff is not None but topk is"
            assert self.topk_recon_coeff is None, "topk_recon_coeff is not None but topk is"

        # Give a warning if both out_recon_coeff and param_match_coeff are > 0
        if (
            self.param_match_coeff is not None
            and self.param_match_coeff > 0
            and self.out_recon_coeff is not None
            and self.out_recon_coeff > 0
        ):
            logger.warning(
                "Both param_match_coeff and out_recon_coeff are > 0. It's typical to only set one."
            )

        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.topk_l2_coeff == 0:
            logger.warning(f"topk_l2_coeff {msg}")
        if self.topk_recon_coeff == 0:
            logger.warning(f"topk_recon_coeff {msg}")
        if self.lp_sparsity_coeff == 0:
            logger.warning(f"lp_sparsity_coeff {msg}")
        if self.param_match_coeff == 0:
            logger.warning(f"param_match_coeff {msg}")

        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert (
                self.lr_exponential_halflife is not None
            ), "lr_exponential_halflife must be set if lr_schedule is exponential"

        if self.full_rank:
            assert not self.unit_norm_matrices, "Can't unit norm matrices if full rank"

        if self.ablation_attributions:
            assert self.topk is not None, "ablation_attributions is only compatible with topk"
        return self


def get_lr_schedule_fn(
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"],
    lr_exponential_halflife: PositiveFloat | None = None,
) -> Callable[[int, int], float]:
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    elif lr_schedule == "exponential":
        assert lr_exponential_halflife is not None  # Should have been caught by model validator
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def get_step_pnorm(step: int, total_steps: int, pnorm_end: float | None = None) -> float:
    if pnorm_end is None:
        return 1.0
    progress = step / total_steps
    return 1 + (pnorm_end - 1) * progress


def get_sparsity_coeff_linear_warmup(
    step: int, steps: int, max_sparsity_coeff: float, sparsity_warmup_pct: float
) -> float:
    warmup_steps = int(steps * sparsity_warmup_pct)
    if step < warmup_steps:
        return max_sparsity_coeff * (step / warmup_steps)
    return max_sparsity_coeff


def get_lr_with_warmup(
    step: int,
    steps: int,
    lr: float,
    lr_schedule_fn: Callable[[int, int], float],
    lr_warmup_pct: float,
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)


def calc_recon_mse(
    output: Float[Tensor, "... n_features"],
    labels: Float[Tensor, "... n_features"],
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    recon_loss = (output - labels) ** 2
    if recon_loss.ndim == 3:
        assert has_instance_dim
        recon_loss = einops.reduce(recon_loss, "b i f -> i", "mean")
    elif recon_loss.ndim == 2:
        recon_loss = recon_loss.mean()
    else:
        raise ValueError(f"Expected 2 or 3 dims in recon_loss, got {recon_loss.ndim}")
    return recon_loss


def calc_topk_l2_rank_one(
    all_As_and_Bs: list[tuple[Float[Tensor, "d_layer_in k"], Float[Tensor, "k d_layer_out"]]],
    topk_mask: Bool[Tensor, "batch ... k"],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Note that we explicitly write the batch dimension to aid understanding. The einsums
    produce the same operation without it. The ... indicates an optional n_instances dimension.

    Args:
        all_As_and_Bs (list[tuple[Float[Tensor, " ... d_in k"], Float[Tensor, " ... k d_out"]]]):
            The A and B matrices for each layer.
        topk_mask (Bool[Tensor, "batch ... k"]): The topk mask to use for the L2 penalty.
            Will contain an n_instances dimension if the model has an n_instances dimension.

    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance (used in tms and
            deep linear toy models).
    """
    batch_size = topk_mask.shape[0]
    n_instances = topk_mask.shape[1] if topk_mask.ndim == 3 else None
    accumulate_shape = (batch_size,) if n_instances is None else (batch_size, n_instances)

    topk_l2_penalty = torch.zeros(accumulate_shape, device=all_As_and_Bs[0][0].device)
    for A, B in all_As_and_Bs:
        # A: [d_in, k] or [n_instances, d_in, k]
        # B: [k, d_out] or [n_instances, k, d_out]
        # topk_mask: [batch, k] or [batch, n_instances, k]
        A_topk = torch.einsum("...fk,b...k ->b...fk", A, topk_mask)
        AB_topk = torch.einsum("b...fk,...kh->b...fh", A_topk, B)
        topk_l2_penalty = topk_l2_penalty + ((AB_topk) ** 2).mean(dim=(-2, -1))
    # Mean over batch_dim and divide by number of parameter matrices we iterated over
    return topk_l2_penalty.mean(dim=0) / len(all_As_and_Bs)


def calc_topk_l2_full_rank(
    subnetwork_params: list[Float[Tensor, " ... k d_in d_out"]],
    topk_mask: Bool[Tensor, "batch ... k"],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Note that we explicitly write the batch dimension to aid understanding. The einsums
    produce the same operation without it. The ... indicates an optional n_instances dimension.

    Args:
        subnetwork_params (list[Float[Tensor, " ... k d_in d_out"]]): The parameters of the
            subnetworks.
        topk_mask (Bool[Tensor, "batch ... k"]): The topk mask to use for the L2 penalty.
            Will contain an n_instances dimension if the model has an n_instances dimension.

    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance (used in tms and
            deep linear toy models).
    """
    assert len(subnetwork_params) > 0, "No subnetwork parameters provided"

    batch_size = topk_mask.shape[0]
    n_instances = topk_mask.shape[1] if topk_mask.ndim == 3 else None
    accumulate_shape = (batch_size,) if n_instances is None else (batch_size, n_instances)

    topk_mask = topk_mask.to(subnetwork_params[0].dtype)
    topk_l2_penalty = torch.zeros(accumulate_shape, device=subnetwork_params[0].device)
    for subnetwork_param_val in subnetwork_params:
        # subnetwork_param: [k, d_in, d_out] or [n_instances, k, d_in, d_out]
        # topk_mask: [batch, k] or [batch, n_instances, k]
        topk_params = einops.einsum(
            subnetwork_param_val, topk_mask, "... k d_in d_out, batch ... k -> batch ... d_in d_out"
        )
        topk_l2_penalty = topk_l2_penalty + ((topk_params) ** 2).mean(dim=(-2, -1))
    # Mean over batch_dim and divide by number of parameter matrices we iterated over
    # NOTE: Assumes all subnetwork params are the same shape, which will not be true if we add
    # biases
    return topk_l2_penalty.mean(dim=0) / len(subnetwork_params)


def calc_param_match_loss(
    pretrained_weights: dict[str, Float[Tensor, " ... d_in d_out"]],
    subnetwork_params_summed: dict[str, Float[Tensor, " ... d_in d_out"]],
    param_map: dict[str, str],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the parameter match loss.

    This is the L2 difference between the combined parameter matrices of the SPDModel and the
    target params.

    Args:
        pretrained_weights (dict[str, Float[Tensor, " ... d_in d_out"]]): The pretrained weights to
            be matched.
        subnetwork_params_summed (dict[str, Float[Tensor, " ... d_in d_out"]]): The parameters of
            the SPDModel (that have already been summed over the subnetwork dimension).
        param_map (dict[str, str]): A map from keys in pretrained_weights to keys in
            subnetwork_params_summed.

    Returns:
        The parameter match loss of shape [n_instances] if the model has an n_instances dimension,
        otherwise of shape [].
    """
    device = next(iter(subnetwork_params_summed.values())).device
    param_match_loss = torch.tensor(0.0, device=device)
    for target_param_name, subnetwork_param_name in param_map.items():
        pretrained_weight = pretrained_weights[target_param_name]
        subnetwork_param = subnetwork_params_summed[subnetwork_param_name]
        param_match_loss = param_match_loss + ((subnetwork_param - pretrained_weight) ** 2).mean(
            dim=(-2, -1)
        )
    return param_match_loss / len(subnetwork_params_summed)


def calc_orthog_loss_full_rank(
    subnetwork_params: list[Float[Tensor, " ... k d_in d_out"]],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the sum of the absolute values of inner products of different subnets.

    NOTE: We could and maybe should try L2 instead of absolute, as well as cosine sim rather than
    dot product.

    Args:
        subnetwork_params (list[Float[Tensor, " ... k d_in d_out"]]): The parameters of the SPDModel

    Returns:
        The orthogonality loss of shape [n_instances] if the model has an n_instances dimension,
        otherwise of shape [].
    """
    first_param = subnetwork_params[0]
    # NOTE: The below assumes that the last three dims are the k, d_in, d_out and will not work
    # for decomposing e.g. biases.
    batch_dims = first_param.shape[:-3]  # All dimensions except the last three
    k = first_param.shape[-3]
    dot_prods = torch.zeros((*batch_dims, k, k), device=first_param.device)
    for subnet in subnetwork_params:
        dot_prods += einops.einsum(
            subnet, subnet, "... k1 d_in d_out, ... k2 d_in d_out -> ... k1 k2"
        )

    # Multiply the k l diagonal by 0
    dot_prods.diagonal(dim1=-2, dim2=-1).zero_()
    orthog_loss = (dot_prods.abs()).mean(dim=(-2, -1))
    return orthog_loss


def calc_lp_sparsity_loss_rank_one(
    out: Float[Tensor, "... d_model_out"],
    layer_acts: list[Float[Tensor, "... d_in"]],
    inner_acts: list[Float[Tensor, "... d_in"]],
    layer_out_params: list[Float[Tensor, "... k d_out"]],
    step_pnorm: float,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Lp sparsity loss on the attributions (inner_acts * d(out)/d(inner_acts).

    Unlike the attributions we calculate for topk in `spd.utils.calc_attributions`, in this function
    we calculate the derivative w.r.t. the layer activations and multiply by that layer's B matrix.
    This will give the same gradient as taking the derivative w.r.t. the inner_acts using the chain
    rule, but importantly it puts the B matrix in the computational graph for this calculation so
    backprop can pass through it (autograd.grad will not build a computational graph from
    intermediate tensors
    https://gist.github.com/danbraunai-apollo/388c3c76be92922cf7b2a2f7da7d0d43). This is a
    (somewhat arbitrary) decision to include this layer's B matrix but not future layer parameters
    in the sparsity loss. We don't do this in topk because topk isn't a differentiable operation
    anyway.

    Args:
        out (Float[Tensor, "... d_model_out"]): The output of the model.
        layer_acts (list[Float[Tensor, "... d_in"]]): Activations at the output of each layer (i.e.
            after both A and B transformations).
        inner_acts (list[Float[Tensor, "... d_in"]]): The inner acts of the model (i.e.
            the set of subnetwork activations after the A transformation for each parameter matrix).
        layer_out_params (list[Float[Tensor, "... k d_out"]]): The output parameters of each layer.
        step_pnorm (float): The pnorm at the current step.

    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension.
    """
    assert len(layer_acts) == len(inner_acts) == len(layer_out_params)
    attributions = torch.zeros_like(inner_acts[0], requires_grad=True)
    for feature_idx in range(out.shape[-1]):
        grad_layer_acts = torch.autograd.grad(
            out[..., feature_idx].sum(),
            layer_acts,
            retain_graph=True,
        )
        sparsity_inner = torch.zeros_like(attributions, requires_grad=True)
        for param_matrix_idx in range(len(layer_out_params)):
            # h_i * grad_h_i
            sparsity_inner = sparsity_inner + (
                inner_acts[param_matrix_idx]
                * torch.einsum(
                    "...o,...ko->...k",
                    grad_layer_acts[param_matrix_idx].detach(),
                    layer_out_params[param_matrix_idx],
                )
            )

        attributions = attributions + sparsity_inner**2
    attributions = attributions / out.shape[-1]

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss = ((attributions.abs() + 1e-16) ** (step_pnorm * 0.5)).sum(dim=-1)
    lp_sparsity_loss = lp_sparsity_loss.mean(dim=0)  # Mean over batch dim
    return lp_sparsity_loss


def calc_lp_sparsity_loss_full_rank(
    out: Float[Tensor, "... d_model_out"],
    layer_acts: list[Float[Tensor, "... d_out"]],
    inner_acts: list[Float[Tensor, "... k d_out"]],
    step_pnorm: float,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Lp sparsity loss on the attributions (inner_acts * d(out)/d(inner_acts).

    Args:
        out (Float[Tensor, "... d_model_out"]): The output of the model.
        layer_acts (list[Float[Tensor, "... d_out"]]): The activations of each layer.
        inner_acts (list[Float[Tensor, "... k d_out"]]): The activations of each subnetwork.
        step_pnorm (float): The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension.
    """
    attributions = calc_attributions_full_rank(out, inner_acts, layer_acts)

    # Average the attributions over the output dimensions
    d_model_out = out.shape[-1]
    attributions = attributions / d_model_out

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss = ((attributions.abs() + 1e-16) ** (step_pnorm * 0.5)).sum(dim=-1)
    lp_sparsity_loss = lp_sparsity_loss.mean(dim=0)  # Mean over batch dim
    return lp_sparsity_loss


def optimize(
    model: SPDModel | SPDFullRankModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    pretrained_model: Model | None,
    param_map: dict[str, str] | None = None,
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    topk = config.topk
    k = config.task_config.k
    lr = config.lr
    assert k is not None, "k must be provided"

    pretrained_model.to(device=device)
    pretrained_params = pretrained_model.state_dict()
    k_params = {}
    # Only use the model for initialization of the k_params
    model_params = model.state_dict()
    decomposable_params = {
        "mlps.0.input_layer.weight": "mlps.0.linear1.subnetwork_params",
        "mlps.0.output_layer.weight": "mlps.0.linear2.subnetwork_params",
    }
    for key1, key2 in decomposable_params.items():
        shape_pretrained = pretrained_params[key1].shape
        shape_model = model_params[key2].shape
        desired_shape = [k, *shape_pretrained]
        print(f"Transposing {key2}: {shape_model} -> {desired_shape}")
        k_params[key1] = model_params[key2].transpose(-2, -1)

    opt = torch.optim.AdamW(k_params.values(), lr=lr, weight_decay=0.0)
    alpha = torch.ones(k, device=device)

    for batch in tqdm(dataloader):
        batch = batch[0].to(device=device)
        # print("Running pretrained model")
        pretrained_out = pretrained_model(batch)

        # print("Compiling test function")

        def calc_jacobian(alpha: Float[Tensor, "k"]) -> Float[Tensor, "batch n_outputs k"]:
            return torch.autograd.functional.jacobian(
                lambda alpha: functional_call(
                    pretrained_model,
                    {k: einops.einsum(v, alpha, "k ..., k -> ...") for k, v in k_params.items()},
                    batch,
                    # is it bad that function doesn't bind loop variable batch? I don't think so
                ),
                alpha,
            )

        # calc_jacobian_compiled = torch.compile(calc_jacobian)
        jacobian = calc_jacobian(alpha)
        # print(f"Jacobian: {(jacobian**2).sum()}")

        # def test_func(alpha):
        #     return functional_call(
        #         pretrained_model,
        #         {k: einops.einsum(v, alpha, "k ..., k -> ...") for k, v in k_params.items()},
        #         batch,
        #     )

        # opt_test_func = torch.compile(test_func)
        # print("Calculating jacobian")
        # jacobian = torch.autograd.functional.jacobian(
        #     opt_test_func,
        #     alpha,
        # ).squeeze(dim=-2)

        # jacobian = torch.autograd.functional.jacobian(
        #     lambda alpha: functional_call(
        #         pretrained_model,
        #         {k: einops.einsum(v, alpha, "k ..., k -> ...") for k, v in k_params.items()},
        #         batch,
        #     ),
        #     alpha,
        # ).squeeze(dim=-2)
        # print("Calculating topk mask")
        attribs: Float[Tensor, "batch k"] = einops.reduce(
            jacobian**2, "batch n_outputs k -> batch k", "sum"
        )
        topk_mask = calc_topk_mask(attribs, topk, batch_topk=True).float()
        print(f"Attribs: {attribs.sum()}")

        # print("Calculating per sample topk forward")

        def per_sample_topk_forward(
            batch_i: Float[Tensor, " n_inputs"],
            topk_mask_i: Float[Tensor, " key"],
            k_params: dict[str, Float[Tensor, " ... k"]],
        ):
            masked_params = {
                key: einops.einsum(value, topk_mask_i, "k ..., k -> ...")
                for key, value in k_params.items()
            }
            return functional_call(pretrained_model, masked_params, batch_i)

        per_sample_topk_forward_p = partial(per_sample_topk_forward, k_params=k_params)
        out_recon = torch.vmap(per_sample_topk_forward_p)(batch, topk_mask)
        recon_loss = (out_recon - pretrained_out).pow(2).mean()

        param_match_loss = 0.0
        for key, value in k_params.items():
            param_match_loss += (value.sum(dim=0) - pretrained_params[key]).pow(2).mean()

        l2_loss = 0.0
        for _, value in k_params.items():
            l2_loss += (
                einops.einsum(
                    topk_mask,
                    value,
                    "b k, k ... -> b k ...",  # could save some memory here by summing
                )
                .pow(2)
                .mean()
            )

        print(
            f"param_match_loss: {param_match_loss: .3e}, l2_loss: {l2_loss: .3e}, recon_loss: {recon_loss: .3e}"
        )
        loss = recon_loss + param_match_loss + l2_loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"Loss: {loss.item()} Attribs: {attribs.sum()}")

"""Run SPD on a model."""

from collections.abc import Callable
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.attributions import calculate_attributions
from spd.configs import Config
from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.module_utils import collect_nested_module_attrs, get_nested_module_attr
from spd.utils import calc_recon_mse, calc_topk_mask, get_lr_schedule_fn, get_lr_with_warmup


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    if config.pnorm is not None:
        run_suffix += f"p{config.pnorm:.2e}_"
    if config.lp_sparsity_coeff is not None:
        run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
    if config.topk is not None:
        run_suffix += f"topk{config.topk:.2e}_"
    if config.topk_recon_coeff is not None:
        run_suffix += f"topkrecon{config.topk_recon_coeff:.2e}_"
    if config.schatten_pnorm is not None:
        run_suffix += f"schatp{config.schatten_pnorm:.2e}_"
    if config.schatten_coeff is not None:
        run_suffix += f"schatten{config.schatten_coeff:.2e}_"
    if config.act_recon_coeff is not None:
        run_suffix += f"actrecon_{config.act_recon_coeff:.2e}_"
    run_suffix += f"C{config.C}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"attr-{config.attribution_type[:3]}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    return run_suffix


def calc_schatten_loss(
    As: dict[str, Float[Tensor, "C d_layer_in m"] | Float[Tensor, "n_instances C d_layer_in m"]],
    Bs: dict[str, Float[Tensor, "C m d_layer_out"] | Float[Tensor, "n_instances C m d_layer_out"]],
    mask: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"],
    p: float,
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Schatten p-norms of the topk subnetworks and sum them.

    Args:
        As: Dictionary of A matrices for each layer
        Bs: Dictionary of B matrices for each layer
        mask: The mask to use for the Schatten p-norm penalty. May be a binary mask (if topk) or
            a float mask (if lp sparsity).
        p: The Schatten p-norm to use (from config.schatten_pnorm)
        n_params: The number of parameters in the model
        device: The device to use for calculations
    Returns:
        The Schatten p-norm penalty for the topk subnetworks
    """
    assert As.keys() == Bs.keys(), "As and Bs must have the same keys"
    n_instances = mask.shape[1] if mask.ndim == 3 else None
    accumulate_shape = (n_instances,) if n_instances is not None else ()

    schatten_penalty = torch.zeros(accumulate_shape, device=device)
    batch_size = mask.shape[0]

    for name in As:
        A = As[name]  # [C, d_in, m] or [n_instances, C, d_in, m]
        B = Bs[name]  # [C, m, d_out] or [n_instances, C, m, d_out]
        # mask: [batch, C] or [batch, n_instances, C]

        # Compute S_A = A^T A and S_B = B B^T
        S_A = einops.einsum(A, A, "... C d_in m, ... C d_in m -> ... C m")
        S_B = einops.einsum(B, B, "... C m d_out, ... C m d_out -> ... C m")

        S_AB = S_A * S_B

        # Apply topk mask
        S_AB_topk = einops.einsum(S_AB, mask, "... C m, batch ... C -> batch ... C m")

        # Sum the Schatten p-norm
        schatten_penalty = schatten_penalty + ((S_AB_topk + 1e-16) ** (0.5 * p)).sum(
            dim=(0, -2, -1)
        )

    return schatten_penalty / n_params / batch_size


def _calc_param_mse(
    params1: dict[str, Float[Tensor, "d_in d_out"] | Float[Tensor, "n_instances d_in d_out"]],
    params2: dict[str, Float[Tensor, "d_in d_out"] | Float[Tensor, "n_instances d_in d_out"]],
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    param_match_loss = torch.tensor(0.0, device=device)
    for name in params1:
        param_match_loss = param_match_loss + ((params2[name] - params1[name]) ** 2).sum(
            dim=(-2, -1)
        )
    return param_match_loss / n_params


def calc_param_match_loss(
    param_names: list[str],
    target_model: HookedRootModule,
    spd_model: SPDModel,
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the MSE between the target model weights and the SPD model weights.

    Args:
        param_names: The names of the parameters to be matched.
        target_model: The target model to match.
        spd_model: The SPD model to match.
        n_params: The number of parameters in the model. Used for normalization.
        device: The device to use for calculations.
    """
    target_params = {}
    spd_params = {}
    for param_name in param_names:
        target_params[param_name] = get_nested_module_attr(target_model, param_name + ".weight")
        spd_params[param_name] = get_nested_module_attr(spd_model, param_name + ".weight")
    return _calc_param_mse(
        params1=target_params,
        params2=spd_params,
        n_params=n_params,
        device=device,
    )


def calc_lp_sparsity_loss(
    out: Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"],
    attributions: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"],
    step_pnorm: float,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the Lp sparsity loss on the attributions.

    Args:
        out: The output of the model.
        attributions: The attributions to use for the sparsity loss.
        step_pnorm: The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension. Note that we keep the batch and C dimensions as we need them if calculating
            the schatten loss.
    """
    # Average the attributions over the output dimensions
    d_model_out = out.shape[-1]
    attributions = attributions / d_model_out

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss_per_k = (attributions.abs() + 1e-16) ** (step_pnorm * 0.5)
    return lp_sparsity_loss_per_k


def calc_act_recon(
    target_post_weight_acts: dict[
        str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]
    ],
    layer_acts: dict[str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """MSE between all target model activations and the output of each subnetwork in the SPD model.

    Args:
        target_post_weight_acts: The activations after each layer in the target model.
        layer_acts: The activations after each subnetwork in the SPD model.

    Returns:
        The activation reconstruction loss. Will have an n_instances dimension if the model has an
            n_instances dimension, otherwise a scalar.
    """
    assert (
        target_post_weight_acts.keys() == layer_acts.keys()
    ), f"Layer keys must match: {target_post_weight_acts.keys()} != {layer_acts.keys()}"

    device = next(iter(layer_acts.values())).device

    total_act_dim = 0  # Accumulate the d_out over all layers for normalization
    loss = torch.zeros(1, device=device)
    for layer_name in target_post_weight_acts:
        total_act_dim += target_post_weight_acts[layer_name].shape[-1]

        error = ((target_post_weight_acts[layer_name] - layer_acts[layer_name]) ** 2).sum(dim=-1)
        loss = loss + error

    # Normalize by the total number of output dimensions and mean over the batch dim
    return (loss / total_act_dim).mean(dim=0)


def optimize(
    model: SPDModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    target_model: HookedRootModule,
    param_names: list[str],
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    model.to(device=device)
    target_model.to(device=device)

    has_instance_dim = hasattr(model, "n_instances")

    # Note that we expect weight decay to be problematic for spd
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    n_params = 0
    for param_name in param_names:
        n_params += get_nested_module_attr(target_model, param_name + ".weight").numel()

    if has_instance_dim:
        # All subnetwork param have an n_instances dimension
        n_params = n_params / model.n_instances

    epoch = 0
    total_samples = 0
    data_iter = iter(dataloader)
    for step in tqdm(range(config.steps + 1), ncols=0):
        if config.unit_norm_matrices:
            assert isinstance(model, SPDModel), "Can only norm matrices in SPDModel instances"
            model.set_As_to_unit_norm()

        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        for group in opt.param_groups:
            group["lr"] = step_lr

        opt.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)[0]  # Ignore labels here, we use the output of target_model
        except StopIteration:
            tqdm.write(f"Epoch {epoch} finished, starting new epoch")
            epoch += 1
            data_iter = iter(dataloader)
            batch = next(data_iter)[0]

        batch = batch.to(device=device)
        total_samples += batch.shape[0]

        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(
            batch, names_filter=target_cache_filter
        )

        # Do a forward pass with all subnetworks
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        out, spd_cache = model.run_with_cache(batch, names_filter=spd_cache_filter)

        # Calculate losses
        out_recon_loss = calc_recon_mse(out, target_out, has_instance_dim)

        param_match_loss = None
        if config.param_match_coeff is not None:
            param_match_loss = calc_param_match_loss(
                param_names=param_names,
                target_model=target_model,
                spd_model=model,
                n_params=n_params,
                device=device,
            )

        post_weight_acts = {k: v for k, v in target_cache.items() if k.endswith("hook_post")}
        pre_weight_acts = {k: v for k, v in target_cache.items() if k.endswith("hook_pre")}
        attributions = calculate_attributions(
            model=model,
            config=config,
            batch=batch,
            out=out,
            target_out=target_out,
            pre_weight_acts=pre_weight_acts,
            post_weight_acts=post_weight_acts,
            component_acts={
                k: v for k, v in spd_cache.items() if k.endswith("hook_component_acts")
            },
        )

        lp_sparsity_loss_per_k = None
        if config.lp_sparsity_coeff is not None:
            assert config.pnorm is not None, "pnorm must be set if lp_sparsity_coeff is set"
            lp_sparsity_loss_per_k = calc_lp_sparsity_loss(
                out=out, attributions=attributions, step_pnorm=config.pnorm
            )

        (
            out_masked,
            schatten_loss,
            masked_recon_loss,
            mask,
            layer_acts_masked,
        ) = None, None, None, None, None
        if config.topk is not None:
            # We always assume the final subnetwork is the one we want to distil
            topk_attrs: Float[Tensor, "batch ... C"] = (
                attributions[..., :-1] if config.distil_from_target else attributions
            )
            if config.exact_topk:
                # Currently only valid for batch_topk and n_instances = 1. Would need to change the
                # topk argument in calc_topk_mask to allow for tensors if relaxing these constraints
                assert config.batch_topk, "exact_topk only works if batch_topk is True"
                assert (
                    hasattr(model, "n_instances") and model.n_instances == 1
                ), "exact_topk only works if n_instances = 1"
                # Get the exact number of active features over the batch
                exact_topk = ((batch != 0).sum() / batch.shape[0]).item()
                mask = calc_topk_mask(topk_attrs, exact_topk, batch_topk=True)
            else:
                mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)
            if config.distil_from_target:
                # Add back the final subnetwork index to the topk mask and set it to True
                last_subnet_mask = torch.ones(
                    (*mask.shape[:-1], 1), dtype=mask.dtype, device=device
                )
                mask = torch.cat((mask, last_subnet_mask), dim=-1)

            # Do a forward pass with only the topk subnetworks
            out_masked, spd_cache_masked = model.run_with_cache(
                batch, names_filter=spd_cache_filter, mask=mask
            )
            layer_acts_masked = {
                k: v for k, v in spd_cache_masked.items() if k.endswith("hook_post")
            }

            if config.topk_recon_coeff is not None:
                assert out_masked is not None
                masked_recon_loss = calc_recon_mse(out_masked, target_out, has_instance_dim)

        act_recon_loss = None
        if config.act_recon_coeff is not None:
            act_recon_layer_acts = (
                layer_acts_masked
                if layer_acts_masked is not None
                else {k: v for k, v in spd_cache.items() if k.endswith("hook_post")}
            )
            target_post_weight_acts = post_weight_acts
            if config.post_relu_act_recon:
                relu = torch.nn.functional.relu
                # Only do post-relu act recon for mlp_in layers and ignore the other layers
                act_recon_layer_acts = {
                    k: relu(v) for k, v in act_recon_layer_acts.items() if "mlp_in" in k
                }
                target_post_weight_acts = {
                    k: relu(v) for k, v in target_post_weight_acts.items() if "mlp_in" in k
                }
            act_recon_loss = calc_act_recon(
                target_post_weight_acts=target_post_weight_acts,
                layer_acts=act_recon_layer_acts,
            )

        if config.schatten_coeff is not None:
            # Use the sparsity loss as the mask in the lp case, and topk_mask otherwise
            mask = mask if mask is not None else lp_sparsity_loss_per_k
            assert mask is not None
            schatten_pnorm = config.schatten_pnorm if config.schatten_pnorm is not None else 1.0
            schatten_loss = calc_schatten_loss(
                As=collect_nested_module_attrs(model, attr_name="A", include_attr_name=False),
                Bs=collect_nested_module_attrs(model, attr_name="B", include_attr_name=False),
                mask=mask,
                p=schatten_pnorm,
                n_params=n_params,
                device=device,
            )

        lp_sparsity_loss = None
        if lp_sparsity_loss_per_k is not None:
            # Sum over the C dimension (-1) and mean over the batch dimension (0)
            lp_sparsity_loss = lp_sparsity_loss_per_k.sum(dim=-1).mean(dim=0)

        loss_terms = {
            "param_match_loss": (param_match_loss, config.param_match_coeff),
            "out_recon_loss": (out_recon_loss, config.out_recon_coeff),
            "lp_sparsity_loss": (lp_sparsity_loss, config.lp_sparsity_coeff),
            "masked_recon_loss": (masked_recon_loss, config.topk_recon_coeff),
            "act_recon_loss": (act_recon_loss, config.act_recon_coeff),
            "schatten_loss": (schatten_loss, config.schatten_coeff),
        }
        # Add up the loss terms
        loss = torch.tensor(0.0, device=device)
        for loss_name, (loss_term, coeff) in loss_terms.items():
            if coeff is not None:
                assert loss_term is not None, f"{loss_name} is None but coeff is not"
                loss = loss + coeff * loss_term.mean()  # Mean over n_instances dimension

        # Logging
        if step % config.print_freq == 0:
            tqdm.write(f"Step {step}")
            tqdm.write(f"Total loss: {loss.item()}")
            tqdm.write(f"lr: {step_lr}")
            for loss_name, (val, _) in loss_terms.items():
                if val is not None:
                    val_repr = f"\n{val.tolist()}" if val.numel() > 1 else f" {val.item()}"
                    tqdm.write(f"{loss_name}:{val_repr}")

            if config.wandb_project:
                metrics = {
                    "pnorm": config.pnorm,
                    "lr": step_lr,
                    "total_loss": loss.item(),
                    **{
                        name: val.mean().item() if val is not None else None
                        for name, (val, _) in loss_terms.items()
                    },
                }
                wandb.log(metrics, step=step)

        # Make plots
        if (
            plot_results_fn is not None
            and config.image_freq is not None
            and step % config.image_freq == 0
            and (step > 0 or config.image_on_first_step)
        ):
            fig_dict = plot_results_fn(
                model=model,
                target_model=target_model,
                step=step,
                out_dir=out_dir,
                device=device,
                config=config,
                topk_mask=mask,
                batch=batch,
            )
            if config.wandb_project:
                wandb.log(
                    {k: wandb.Image(v) for k, v in fig_dict.items()},
                    step=step,
                )

        # Save model
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"spd_model_{step}.pth")
            tqdm.write(f"Saved model to {out_dir / f'spd_model_{step}.pth'}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"spd_model_{step}.pth"), base_path=out_dir, policy="now")

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            loss.backward(retain_graph=True)

            if step % config.print_freq == 0 and config.wandb_project:
                # Calculate gradient norm
                grad_norm: float = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm()  # type: ignore
                wandb.log({"grad_norm": grad_norm}, step=step)

            if config.unit_norm_matrices:
                model.fix_normalized_adam_gradients()

            opt.step()

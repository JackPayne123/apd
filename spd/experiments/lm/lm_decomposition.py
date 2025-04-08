"""Language Model decomposition script."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import einops
import fire
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import wandb
import yaml
from jaxtyping import Float
from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config, LMTaskConfig
from spd.experiments.lm.models import (
    LinearComponentWithBias,
    SSModel,
)
from spd.log import logger
from spd.models.components import Gate, GateMLP
from spd.run_spd import _calc_param_mse, calc_masks, calc_random_masks, get_common_run_name_suffix
from spd.utils import (
    get_device,
    get_lr_schedule_fn,
    get_lr_with_warmup,
    load_config,
    set_seed,
)
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(
    config: Config,
    model_size: str,
    max_seq_len: int,
) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"_lm{model_size}_seq{max_seq_len}"
    return config.wandb_run_name_prefix + run_suffix


def lm_plot_results_fn(
    model: SSModel,
    components: dict[str, LinearComponentWithBias],
    step: int | None,
    out_dir: Path | None,
    device: str,
    config: Config,
    **_,
) -> dict[str, plt.Figure]:
    """Plotting function for LM decomposition. Placeholder for now."""
    # TODO: Implement actual plotting (e.g., component matrix values?)
    logger.info(f"Plotting results at step {step}...")
    fig_dict: dict[str, plt.Figure] = {}
    # Example: Potentially plot A/B matrix norms or sparsity patterns?
    # fig_dict["component_norms"] = plot_component_norms(components, out_dir, step)
    return fig_dict


def calc_component_acts(
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
    As: dict[str, Float[nn.Parameter, "d_in m"]],
) -> dict[str, Float[Tensor, "batch m"]]:
    """Calculate the component acts for each layer. I.e. (pre_weight_acts @ A).

    Args:
        pre_weight_acts: The activations before each layer in the target model.
        As: The A matrix at each layer.
    """
    component_acts = {}
    for param_name in pre_weight_acts:
        component_acts[param_name] = einops.einsum(
            pre_weight_acts[param_name], As[param_name], "... d_in, ... d_in m -> ... m"
        )
    return component_acts


def calc_recon_mse_lm(
    out1: Float[Tensor, "batch pos vocab"],
    out2: Float[Tensor, "batch pos vocab"],
) -> Float[Tensor, ""]:
    """Calculate the Mean Squared Error reconstruction loss for LM logits."""
    assert out1.shape == out2.shape
    # Mean over batch and sequence length, sum over vocab
    return ((out1 - out2) ** 2).sum(dim=-1).mean()


def calc_param_match_loss_lm(
    components: dict[str, LinearComponentWithBias],
    target_model: Llama,
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE loss between component parameters (A@B + bias) and target parameters."""
    target_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    component_params: dict[str, Float[Tensor, "d_in d_out"]] = {}

    for comp_name, component in components.items():
        component_params[comp_name] = einops.einsum(
            component.linear_component.A,
            component.linear_component.B,
            "d_in m, m d_out -> d_in d_out",
        )
        target_params[comp_name] = target_model.get_parameter(comp_name + ".weight").T
        assert component_params[comp_name].shape == target_params[comp_name].shape

    param_mse = _calc_param_mse(
        params1=component_params,
        params2=target_params,
        n_params=n_params,
        device=device,
    )
    return param_mse


def calc_layerwise_recon_loss_lm(
    model: SSModel,
    batch: Float[Tensor, "batch pos"],
    device: str,
    masks: list[dict[str, Float[Tensor, "batch pos m"]]],
    target_out: Float[Tensor, "batch pos vocab"],
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component at a time."""
    n_modified_components = len(masks[0])
    total_loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        for module_name in mask_info:
            modified_out, _ = model.forward_with_component(
                batch, module_name=module_name, mask=mask_info[module_name]
            )
            loss = calc_recon_mse_lm(modified_out, target_out)
            total_loss += loss
    return total_loss / (n_modified_components * len(masks))


def calc_lp_sparsity_loss_lm(
    relud_masks: dict[str, Float[Tensor, "batch pos m"]], pnorm: float
) -> Float[Tensor, ""]:
    """Calculate the Lp sparsity loss on the attributions.

    Args:
        relud_masks: Dictionary of relu masks for each layer.
        pnorm: The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss.
    """
    # Initialize with zeros matching the shape of first mask
    total_loss = torch.zeros_like(next(iter(relud_masks.values())))

    for layer_relud_mask in relud_masks.values():
        total_loss = total_loss + layer_relud_mask**pnorm

    # Sum over the m dimension and mean over the batch and pos dimensions
    return total_loss.sum(dim=-1).mean(dim=[0, 1])


def optimize_lm(
    model: SSModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "batch pos"], Float[Tensor, "batch pos"]]],
    plot_results_fn: Callable[..., dict[str, plt.Figure]],
    out_dir: Path | None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore

    component_params = []
    param_names_to_optimize = []
    for name, component in model.components.items():
        component_params.extend(list(component.parameters()))
        param_names_to_optimize.extend(
            [f"{name}.{p_name}" for p_name, _ in component.named_parameters()]
        )
        logger.debug(f"Adding parameters from component: {name}")

    if not component_params:
        logger.error("No parameters found in components to optimize. Exiting.")
        return

    optimizer = optim.AdamW(component_params, lr=config.lr, weight_decay=0.0)
    logger.info(f"Optimizer created for params: {param_names_to_optimize}")

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    n_params = 0
    for module_name in model.components:
        weight = model.model.get_parameter(module_name + ".weight")
        n_params += weight.numel()

    log_data = {}
    data_iter = iter(dataloader)

    # Use tqdm directly in the loop, iterate one extra step for final logging/plotting/saving
    for step in tqdm(range(config.steps + 1), ncols=0):
        # --- LR Scheduling Step --- #
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        # Manually update optimizer's learning rate
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        log_data["lr"] = step_lr

        # --- Zero Gradients --- #
        optimizer.zero_grad()

        # --- Get Batch --- #
        try:
            batch = next(data_iter)["input_ids"].to(device)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            data_iter = iter(dataloader)
            batch = next(data_iter)["input_ids"].to(device)

        (target_out, _), pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(model.components.keys())
        )
        As = {module_name: v.linear_component.A for module_name, v in model.components.items()}

        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)
        # attributions = calc_grad_attributions(
        #     target_out=target_out,
        #     pre_weight_acts=pre_weight_acts,
        #     post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
        #     target_component_acts=target_component_acts,
        #     Bs=collect_nested_module_attrs(model, attr_name="B", include_attr_name=False),
        # )
        attributions = None

        masks, relud_masks = calc_masks(
            gates=gates,
            target_component_acts=target_component_acts,
            attributions=attributions,
            detach_inputs=False,
        )

        # --- Calculate Losses --- #
        total_loss = torch.tensor(0.0, device=device)
        loss_terms = {}

        ####### param match loss #######
        param_match_loss_val = calc_param_match_loss_lm(
            components=model.components,
            target_model=model.model,
            n_params=n_params,
            device=device,
        )
        total_loss += config.param_match_coeff * param_match_loss_val
        loss_terms["loss/parameter_matching"] = param_match_loss_val.item()

        ####### layerwise recon loss #######
        if config.layerwise_recon_coeff is not None:
            layerwise_recon_loss = calc_layerwise_recon_loss_lm(
                model=model,
                batch=batch,
                device=device,
                masks=[masks],
                target_out=target_out,
            )
            total_loss += config.layerwise_recon_coeff * layerwise_recon_loss
            loss_terms["loss/layerwise_reconstruction"] = layerwise_recon_loss.item()

        ####### layerwise random recon loss #######
        if config.layerwise_random_recon_coeff is not None:
            layerwise_random_masks = calc_random_masks(
                masks=masks, n_random_masks=config.n_random_masks
            )
            layerwise_random_recon_loss = calc_layerwise_recon_loss_lm(
                model=model,
                batch=batch,
                device=device,
                masks=layerwise_random_masks,
                target_out=target_out,
            )
            total_loss += config.layerwise_random_recon_coeff * layerwise_random_recon_loss
            loss_terms["loss/layerwise_random_reconstruction"] = layerwise_random_recon_loss.item()

        ####### lp sparsity loss #######
        lp_sparsity_loss = calc_lp_sparsity_loss_lm(relud_masks=relud_masks, pnorm=config.pnorm)
        total_loss += config.lp_sparsity_coeff * lp_sparsity_loss
        loss_terms["loss/lp_sparsity_loss"] = lp_sparsity_loss.item()

        ####### out recon loss #######
        if config.out_recon_coeff is not None:
            # Get target logits (no gradients needed for target model)
            with torch.no_grad():
                target_logits, _ = model.forward(batch)
                # Detach target logits to ensure no grads flow back
                target_logits = target_logits.detach()

            # Get component logits
            component_logits, _ = model.forward_with_components(batch, masks=masks)

            assert component_logits.shape == target_logits.shape, (
                f"Shape mismatch: {component_logits.shape} vs {target_logits.shape}"
            )

            recon_loss = calc_recon_mse_lm(component_logits, target_logits)
            total_loss += config.out_recon_coeff * recon_loss
            loss_terms["loss/reconstruction"] = recon_loss.item()

        # # --- Placeholder Losses (Mimicking run_spd.optimize structure) ---
        # masked_recon_loss_val = None
        # if config.masked_recon_coeff is not None and config.masked_recon_coeff > 0:
        #     logger.warning("masked_recon_loss requires mask calculation implementation.")
        #     # TODO: Calculate masked_recon_loss_val using masks
        #     # e.g., component_logits_masked = model.forward_with_components(..., masks=masks)
        #     #       masked_recon_loss_val = calc_recon_mse_lm(component_logits_masked, target_logits)
        #     loss_terms["loss/masked_reconstruction"] = None  # Or 0.0 if calculated

        # act_recon_loss_val = None
        # if config.act_recon_coeff is not None and config.act_recon_coeff > 0:
        #     logger.warning("act_recon_loss requires mask and target activation calculation.")
        #     # TODO: Implement act_recon_loss_val
        #     loss_terms["loss/activation_reconstruction"] = None

        # random_masks_loss_val = None
        # if config.random_mask_recon_coeff is not None and config.random_mask_recon_coeff > 0:
        #     logger.warning("random_masks_loss requires mask calculation implementation.")
        #     # TODO: Implement random_masks_loss_val
        #     loss_terms["loss/random_mask_reconstruction"] = None

        # layerwise_recon_loss_val = None
        # if config.layerwise_recon_coeff is not None and config.layerwise_recon_coeff > 0:
        #     logger.warning("layerwise_recon_loss requires mask calculation and layerwise hooks.")
        #     # TODO: Implement layerwise_recon_loss_val
        #     loss_terms["loss/layerwise_reconstruction"] = None

        # layerwise_random_recon_loss_val = None
        # if (
        #     config.layerwise_random_recon_coeff is not None
        #     and config.layerwise_random_recon_coeff > 0
        # ):
        #     logger.warning(
        #         "layerwise_random_recon_loss requires mask calculation and layerwise hooks."
        #     )
        #     # TODO: Implement layerwise_random_recon_loss_val
        #     loss_terms["loss/layerwise_random_reconstruction"] = None

        log_data["loss/total"] = total_loss.item()
        log_data.update(loss_terms)

        # --- Logging --- #
        if step % config.print_freq == 0:
            tqdm.write(f"--- Step {step} ---")
            tqdm.write(f"LR: {step_lr:.6f}")
            tqdm.write(f"Total Loss: {log_data['loss/total']:.7f}")
            for name, value in loss_terms.items():
                if value is not None:
                    tqdm.write(f"{name}: {value:.7f}")

            if config.wandb_project:
                wandb.log(log_data, step=step)

        # --- Plotting --- #
        if (
            config.image_freq is not None
            and step % config.image_freq == 0
            and (step > 0 or config.image_on_first_step)
        ):
            logger.info(f"Step {step}: Generating plots...")
            with torch.no_grad():
                fig_dict = plot_results_fn(
                    model=model,  # Pass the SSModel wrapper
                    components=model.components,
                    step=step,
                    out_dir=out_dir,
                    device=device,
                    config=config,
                    # Add any other necessary args for plotting like tokenizer, sample text?
                )
                if config.wandb_project:
                    wandb.log(
                        {k: wandb.Image(v) for k, v in fig_dict.items()},
                        step=step,
                    )
                    if out_dir is not None:
                        for k, v in fig_dict.items():
                            v.savefig(out_dir / f"{k}_{step}.png")
                            tqdm.write(f"Saved plot to {out_dir / f'{k}_{step}.png'}")

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            checkpoint_dir = out_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f"components_step_{step}.pt"
            # Save only component state dicts
            component_state_dicts = {n: c.state_dict() for n, c in model.components.items()}
            save_payload = {
                "components": component_state_dicts,
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": config.model_dump(mode="json"),
            }
            torch.save(save_payload, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            if config.wandb_project:
                wandb.save(str(checkpoint_path), base_path=str(out_dir), policy="now")

        # --- Backward Pass & Optimize --- #
        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            total_loss.backward(retain_graph=True)

            if step % config.print_freq == 0 and config.wandb_project:
                # Calculate gradient norm
                grad_norm: float = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm()  # type: ignore
                wandb.log({"grad_norm": grad_norm}, step=step)

            if config.unit_norm_matrices:
                model.fix_normalized_adam_gradients()

            optimizer.step()
    logger.info("Finished training loop.")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, LMTaskConfig), (
        "Task config must be LMTaskConfig for LM decomposition."
    )

    # --- Load Model --- #
    logger.info(f"Loading model: {config.task_config.model_size}")
    model_config_dict = MODEL_CONFIGS[config.task_config.model_size]
    model_path = f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}"
    model = Llama.from_pretrained(model_path, model_config_dict)

    ss_model = SSModel(
        llama_model=model,
        target_module_patterns=config.task_config.target_module_patterns,
        m=config.m,
        n_gate_hidden_neurons=config.n_gate_hidden_neurons,
    )
    ss_model.to(device)
    logger.info("Model loaded.")

    # --- Setup Run Name and Output Dir --- #
    run_name = get_run_name(
        config,
        model_size=config.task_config.model_size,
        max_seq_len=config.task_config.max_seq_len,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- Save Config --- #
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    # --- Load Data --- #
    logger.info("Loading dataset...")
    dataset_config = DatasetConfig(
        name=config.task_config.dataset_name,
        tokenizer_file_path=None,
        hf_tokenizer_path=model_path,
        split=config.task_config.dataset_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name="story",
    )

    dataloader, tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )
    logger.info("Dataset and tokenizer loaded.")

    logger.info("Freezing target model parameters...")
    for param in ss_model.model.parameters():
        param.requires_grad = False
    logger.info("Target model frozen.")

    logger.info("Starting optimization...")
    optimize_lm(
        model=ss_model,
        config=config,
        device=device,
        dataloader=dataloader,
        out_dir=out_dir,
        plot_results_fn=lm_plot_results_fn,
    )

    logger.info("Optimization finished.")

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

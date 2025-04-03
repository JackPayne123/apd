"""Language Model decomposition script."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

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
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from spd.configs import Config, LMTaskConfig
from spd.experiments.lm.models import (
    LinearComponentWithBias,
    SSModel,
    create_target_components,
)
from spd.log import logger
from spd.run_spd import get_common_run_name_suffix
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


def calc_recon_mse_lm(
    out1: Float[Tensor, "batch seq vocab"],
    out2: Float[Tensor, "batch seq vocab"],
) -> Float[Tensor, ""]:
    """Calculate the Mean Squared Error reconstruction loss for LM logits."""
    assert out1.shape == out2.shape
    # Mean over batch and sequence length, sum over vocab
    return ((out1 - out2) ** 2).sum(dim=-1).mean()


def optimize_lm(
    model: SSModel,
    components: dict[str, LinearComponentWithBias],
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "batch pos"], Float[Tensor, "batch pos"]]],
    out_dir: Path,
    plot_results_fn: Callable[..., dict[str, plt.Figure]],
) -> None:
    """Run the optimization loop for LM decomposition."""
    # --- Optimizer --- #
    component_params = []
    param_names_to_optimize = []
    for name, component in components.items():
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
    logger.info(f"Optimizer details: {optimizer}")

    # --- Scheduler --- #
    # Get the base LR schedule function (e.g., constant, linear, cosine)
    lr_schedule_fn = get_lr_schedule_fn(
        config.lr_schedule,
        config.lr_exponential_halflife,
    )
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    # --- Training Loop --- #
    pbar = tqdm(range(config.steps), desc="Optimizing Components")
    log_data = {}
    # Make dataloader an iterator
    # TODO: Handle dataloader exhaustion if it's finite (e.g., for validation)
    data_iter = iter(dataloader)

    for step in pbar:
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
            batch = next(data_iter)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)

        # --- Calculate Losses --- #
        total_loss = torch.tensor(0.0, device=device)
        loss_terms = {}

        # 1. Reconstruction Loss (comparing logits)
        if config.out_recon_coeff is not None and config.out_recon_coeff > 0:
            # Get target logits (no gradients needed for target model)
            with torch.no_grad():
                target_logits, _ = model.forward(input_ids)
                # Detach target logits to ensure no grads flow back
                target_logits = target_logits.detach()

            # Get component logits
            component_logits, _ = model.forward_with_components(input_ids, components=components)

            # Ensure shapes match (Batch, SeqLen-1, VocabSize)
            assert component_logits.shape == target_logits.shape, (
                f"Shape mismatch: {component_logits.shape} vs {target_logits.shape}"
            )

            recon_loss = calc_recon_mse_lm(component_logits, target_logits)
            total_loss += config.out_recon_coeff * recon_loss
            loss_terms["loss/reconstruction"] = recon_loss.item()

        # 2. Sparsity Loss (Lp norm on component parameters)
        # Note: Using p=config.pnorm. The original optimize used relud_masks from gates.
        lp_sparsity_loss_val = None
        if config.lp_sparsity_coeff > 0:
            lp_norm = torch.tensor(0.0, device=device)
            for component in components.values():
                # Apply Lp loss to A and B matrices
                lp_norm += torch.norm(component.linear_component.A, p=config.pnorm)
                lp_norm += torch.norm(component.linear_component.B, p=config.pnorm)

            lp_sparsity_loss_val = lp_norm
            total_loss += config.lp_sparsity_coeff * lp_sparsity_loss_val
            loss_terms[f"loss/sparsity_l{config.pnorm}_params"] = lp_sparsity_loss_val.item()

        # --- Placeholder Losses (Mimicking run_spd.optimize structure) ---
        # These require a mechanism for calculating masks specific to the LM setup.
        masks = None  # Placeholder: Masks are needed for the following losses
        masked_recon_loss_val = None
        if config.masked_recon_coeff is not None and config.masked_recon_coeff > 0:
            logger.warning("masked_recon_loss requires mask calculation implementation.")
            # TODO: Calculate masked_recon_loss_val using masks
            # e.g., component_logits_masked = model.forward_with_components(..., masks=masks)
            #       masked_recon_loss_val = calc_recon_mse_lm(component_logits_masked, target_logits)
            loss_terms["loss/masked_reconstruction"] = None  # Or 0.0 if calculated

        act_recon_loss_val = None
        if config.act_recon_coeff is not None and config.act_recon_coeff > 0:
            logger.warning("act_recon_loss requires mask and target activation calculation.")
            # TODO: Implement act_recon_loss_val
            loss_terms["loss/activation_reconstruction"] = None

        random_masks_loss_val = None
        if config.random_mask_recon_coeff is not None and config.random_mask_recon_coeff > 0:
            logger.warning("random_masks_loss requires mask calculation implementation.")
            # TODO: Implement random_masks_loss_val
            loss_terms["loss/random_mask_reconstruction"] = None

        layerwise_recon_loss_val = None
        if config.layerwise_recon_coeff is not None and config.layerwise_recon_coeff > 0:
            logger.warning("layerwise_recon_loss requires mask calculation and layerwise hooks.")
            # TODO: Implement layerwise_recon_loss_val
            loss_terms["loss/layerwise_reconstruction"] = None

        layerwise_random_recon_loss_val = None
        if (
            config.layerwise_random_recon_coeff is not None
            and config.layerwise_random_recon_coeff > 0
        ):
            logger.warning(
                "layerwise_random_recon_loss requires mask calculation and layerwise hooks."
            )
            # TODO: Implement layerwise_random_recon_loss_val
            loss_terms["loss/layerwise_random_reconstruction"] = None

        # Add placeholder losses to total_loss if they were calculated (currently they are not)
        # Example if masked_recon_loss_val was calculated:
        # if masked_recon_loss_val is not None:
        #     total_loss += config.masked_recon_coeff * masked_recon_loss_val
        # Repeat for other placeholder losses...

        # --- Backward Pass & Optimize --- #
        if total_loss.requires_grad:
            total_loss.backward()
            # Optional: Gradient Clipping
            # grad_norm_clip_val = 1.0
            # grad_norm = torch.nn.utils.clip_grad_norm_(component_params, max_norm=grad_norm_clip_val)
            # log_data["grad_norm/clipped"] = grad_norm.item()

            optimizer.step()
        elif total_loss == 0.0:
            logger.warning(f"Step {step}: Total loss is zero, skipping backward/optimize.")
        else:
            logger.warning(f"Step {step}: No loss requires grad, skipping backward/optimize.")

        log_data["loss/total"] = total_loss.item()

        # --- Logging --- #
        if step % config.print_freq == 0 or step == config.steps - 1:
            log_data.update(loss_terms)  # Add individual loss terms for logging
            pbar.set_postfix(log_data)
            if config.wandb_project:
                wandb.log(log_data, step=step)
            # Reset loss_terms part of log_data for next interval, keep LR
            log_data = {"lr": step_lr}

        # --- Plotting --- #
        if config.image_freq is not None and (
            (
                step % config.image_freq == 0 and step > 0
            )  # Avoid plotting at step 0 unless requested
            or (config.image_on_first_step and step == 0)
            or (step == config.steps - 1)  # Always plot at the end
        ):
            logger.info(f"Step {step}: Generating plots...")
            # Ensure model is in eval mode for plotting if necessary, though shouldn't matter here
            # model.eval()
            with torch.no_grad():
                figures = plot_results_fn(
                    model=model,  # Pass the SSModel wrapper
                    components=components,
                    step=step,
                    out_dir=out_dir,
                    device=device,
                    config=config,
                    # Add any other necessary args for plotting like tokenizer, sample text?
                )
                if config.wandb_project and figures:
                    wandb.log({f"plots/{k}": wandb.Image(v) for k, v in figures.items()}, step=step)
            # model.train() # Set back to train mode if needed

        # --- Saving Checkpoints --- #
        if (config.save_freq is not None and step % config.save_freq == 0 and step > 0) or (
            step == config.steps - 1
        ):
            checkpoint_dir = out_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f"components_step_{step}.pt"
            # Save only component state dicts
            component_state_dicts = {n: c.state_dict() for n, c in components.items()}
            save_payload = {
                "components": component_state_dicts,
                "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict(),
                "step": step,
                "config": config.model_dump(mode="json"),
            }
            torch.save(save_payload, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            if config.wandb_project:
                wandb.save(str(checkpoint_path), base_path=str(out_dir), policy="now")

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
    ss_model = SSModel(model)
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

    # --- Initialize Components --- #
    logger.info(
        f"Initializing components for modules matching: {config.task_config.target_module_patterns}"
    )
    components = create_target_components(
        ss_model.model,
        rank=config.m,
        target_module_patterns=config.task_config.target_module_patterns,
        device=device,
    )
    logger.info(f"Created {len(components)} components: {list(components.keys())}")

    logger.info("Starting optimization...")
    optimize_lm(
        model=ss_model,
        components=components,
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

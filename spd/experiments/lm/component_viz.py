"""
Vizualises the components of the model.
"""

from pathlib import Path

import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import LMTaskConfig
from spd.experiments.lm.lm_decomposition import calc_component_acts
from spd.experiments.lm.models import LinearComponentWithBias, SSModel
from spd.log import logger
from spd.models.components import Gate, GateMLP
from spd.run_spd import calc_masks
from spd.types import ModelPath


def component_activation_statistics(
    model: SSModel,
    dataloader: DataLoader[Float[Tensor, "batch pos"]],
    n_steps: int,
    device: str,
    out_dir: Path,
) -> None:
    """Get the number and strength of the masks over the full dataset."""
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponentWithBias] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    n_tokens = {module_name.replace("-", "."): 0 for module_name in components}
    total_n_active_components = {module_name.replace("-", "."): 0 for module_name in components}
    component_activation_counts = {
        module_name.replace("-", "."): torch.zeros(model.m, device=device)
        for module_name in components
    }
    data_iter = iter(dataloader)
    for _ in tqdm(range(n_steps), ncols=0):
        # --- Get Batch --- #
        batch = next(data_iter)["input_ids"].to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: v.linear_component.A for module_name, v in components.items()}

        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)

        masks, relud_masks = calc_masks(
            gates=gates,
            target_component_acts=target_component_acts,
            attributions=None,
            detach_inputs=False,
        )
        for module_name, mask in masks.items():
            assert mask.ndim == 3  # (batch_size, pos, m)
            n_tokens[module_name] += mask.shape[0] * mask.shape[1]
            # Count the number of components that are active at all
            active_components = mask > 0
            total_n_active_components[module_name] += int(active_components.sum().item())
            component_activation_counts[module_name] += active_components.sum(dim=(0, 1))

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " m"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in components
    }

    logger.info(f"n_components: {model.m}")
    logger.info(f"mean_n_active_components_per_token: {mean_n_active_components_per_token}")
    logger.info(f"mean_component_activation_counts: {mean_component_activation_counts}")
    for module_name, counts in mean_component_activation_counts.items():
        name = module_name.replace(".", "-")
        plt.hist(counts.detach().cpu().numpy(), bins=100)
        plt.savefig(out_dir / f"{name}_mean_component_activation_counts.png")
        print("Saved plot to", out_dir / f"{name}_mean_component_activation_counts.png")


def main(path: ModelPath) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ss_model, config, checkpoint_path = SSModel.from_pretrained(path)
    ss_model.to(device)

    out_dir = checkpoint_path

    assert isinstance(config.task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=config.task_config.dataset_name,
        tokenizer_file_path=None,
        hf_tokenizer_path=f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}",
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
    # print(ss_model)
    print(config)

    component_activation_statistics(
        model=ss_model,
        dataloader=dataloader,
        n_steps=100,
        device=device,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    path = "wandb:spd-lm/runs/fuff71ef"
    main(path)

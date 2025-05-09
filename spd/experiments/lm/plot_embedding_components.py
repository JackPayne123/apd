"""Visualize embedding component masks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.experiments.lm.models import EmbeddingComponent, SSModel
from spd.models.components import Gate, GateMLP
from spd.run_spd import calc_component_acts, calc_masks


def collect_embedding_masks(model: SSModel, device: str) -> Float[Tensor, "vocab m"]:
    """Collect masks for each vocab token.

    Args:
        model: The trained SSModel
        device: Device to run computation on

    Returns:
        Tensor of shape (vocab_size, m) containing masks for each vocab token
    """
    # We used "-" instead ofGateMLP module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    assert len(components) == 1, "Expected exactly one embedding component"
    component_name = next(iter(components.keys()))

    vocab_size = model.model.get_parameter("transformer.wte.weight").shape[0]

    all_masks = torch.zeros((vocab_size, model.m), device=device)

    for token_id in tqdm(range(vocab_size), desc="Collecting masks"):
        # Create single token input
        token_tensor = torch.tensor([[token_id]], device=device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            token_tensor, module_names=[component_name]
        )

        As = {module_name: v.linear_component.A for module_name, v in components.items()}
        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)  # type: ignore

        masks, _ = calc_masks(
            gates=gates,
            target_component_acts=target_component_acts,
            attributions=None,
            detach_inputs=True,
        )

        all_masks[token_id] = masks[component_name].squeeze()

    return all_masks


def plot_embedding_mask_heatmap(masks: Float[Tensor, "vocab m"], out_dir: Path) -> None:
    """Plot heatmap of embedding masks.

    Args:
        masks: Tensor of shape (vocab_size, m) containing masks
        out_dir: Directory to save the plots
    """
    plt.figure(figsize=(20, 10))
    plt.imshow(
        masks.detach().cpu().numpy(),
        aspect="auto",  # Maintain the data aspect ratio
        cmap="Reds",  # white → red
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="Mask value")

    # Set axis ticks
    plt.xticks(range(0, masks.shape[1], 1000))  # Show every 1000th tick on x-axis
    plt.yticks(range(0, masks.shape[0], 1000))  # Show every 1000th tick on y-axis

    plt.xlabel("Component Index (m)")
    plt.ylabel("Vocab Token ID")
    plt.title("Embedding Component Masks per Token")
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_masks.png", dpi=300)
    plt.savefig(out_dir / "embedding_masks.svg")  # vector graphic for zooming
    print(f"Saved embedding masks to {out_dir / 'embedding_masks.png'} and .svg")
    plt.close()

    # Also plot a histogram of the first token's mask
    threshold = 0.05
    indices = [0, 99, 199, 299]
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs = axs.flatten()  # type: ignore
    for token_id, ax in zip(indices, axs, strict=False):
        vals = masks[token_id].detach().cpu().numpy()
        vals = vals[vals > threshold]

        # Ensure all sub-plots have the same ticks and visible range
        ax.set_xticks(np.arange(0.0, 1.05 + 1e-6, 0.05))
        ax.set_xlim(0.0, 1.05)
        ax.hist(vals, bins=100)
        ax.set_ylabel(f"Freq for token {token_id}")

    fig.suptitle(f"Mask Values (> {threshold}) for Each Token")
    plt.savefig(out_dir / "first_token_histogram.png")
    plt.savefig(out_dir / "first_token_histogram.svg")  # vector version
    print(f"Saved first token histogram to {out_dir / 'first_token_histogram.png'} and .svg")
    plt.close()

    n_dead_components = ((masks > 0.1).sum(dim=0) == 0).sum().item()
    print(f"Number of components that have no value > 0.1: {n_dead_components}")
    ...


def main(model_path: str | Path) -> None:
    """Load model and generate embedding mask visualization.

    Args:
        model_path: Path to the model checkpoint
    """
    # Load model
    model, config, out_dir = SSModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Collect masks
    masks = collect_embedding_masks(model, device)

    plot_embedding_mask_heatmap(masks, out_dir)


if __name__ == "__main__":
    # path = "wandb:spd-lm/runs/cllwvnmz" # Run with some components that always activate.
    path = "wandb:spd-lm/runs/6u0i6eax"  # Some components activate 0.175 of the time.

    main(path)

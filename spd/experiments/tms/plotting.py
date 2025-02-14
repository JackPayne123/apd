import einops
import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.experiments.tms.models import TMSModel, TMSSPDModel
from spd.models.components import Gate
from spd.module_utils import collect_nested_module_attrs
from spd.run_spd import calc_component_acts, calc_masks


def plot_mask_vals(
    model: TMSSPDModel,
    target_model: TMSModel,
    gates: dict[str, Gate],
    device: str,
    input_magnitude: float,
) -> plt.Figure:
    """Plot the values of the mask for a batch of inputs with single active features."""
    # First, create a batch of inputs with single active features
    n_features = model.n_features
    n_instances = model.n_instances
    batch = torch.eye(n_features, device=device) * input_magnitude
    batch = einops.repeat(
        batch, "batch n_features -> batch n_instances n_features", n_instances=n_instances
    )

    # Forward pass with target model
    target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
    target_cache = target_model.run_with_cache(batch, names_filter=target_cache_filter)[1]
    pre_weight_acts = {k: v for k, v in target_cache.items() if k.endswith("hook_pre")}
    As = collect_nested_module_attrs(model, attr_name="A", include_attr_name=False)

    target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)

    relud_masks = calc_masks(
        gates=gates, target_component_acts=target_component_acts, attributions=None
    )[1]

    # Create figure with better layout and sizing
    fig, axs = plt.subplots(
        len(relud_masks),
        n_instances,
        figsize=(5 * n_instances, 5 * len(relud_masks)),
        constrained_layout=True,
    )
    axs = np.array(axs)

    images = []
    for i in range(n_instances):
        axs[0, i].set_title(f"Instance {i}")
        for j, (mask_name, mask) in enumerate(relud_masks.items()):
            # mask has shape (batch, n_instances, m)
            mask_data = mask[:, i, :].detach().cpu().numpy()
            im = axs[j, i].matshow(mask_data, aspect="auto", cmap="Reds")
            images.append(im)

            axs[j, i].set_xlabel("Mask index")
            if i == 0:  # Only set ylabel for leftmost plots
                axs[j, i].set_ylabel("Input feature index")
            axs[j, i].set_title(mask_name)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in relud_masks.values()),
        vmax=max(mask.max().item() for mask in relud_masks.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Add a title which shows the input magnitude
    fig.suptitle(f"Input magnitude: {input_magnitude}")

    return fig


# pretrained_model_path = "wandb:spd-train-tms/runs/tmzweoqk"
# run_id = "wandb:spd-tms/runs/7qvf63x8"


# target_model, target_model_train_config_dict = TMSModel.from_pretrained(pretrained_model_path)
# spd_model, spd_model_train_config_dict = TMSSPDModel.from_pretrained(run_id)

# # We used "-" instead of "." as module names can't have "." in them
# gates = {k.removeprefix("gates.").replace("-", "."): v for k, v in spd_model.gates.items()}

# fig = plot_mask_vals(spd_model, target_model, gates, device="cpu", input_magnitude=0.5)
# fig.savefig("tms_mask_vals.png")

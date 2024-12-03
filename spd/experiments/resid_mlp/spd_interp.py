# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import Normalize
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDRankPenaltyModel
from spd.experiments.resid_mlp.plotting import (
    plot_individual_feature_response,
    plot_virtual_weights,
    relu_contribution_plot,
    spd_calculate_virtual_weights,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import ResidualMLPTaskConfig, calc_recon_mse
from spd.utils import run_spd_forward_pass, set_seed

# %% Loading
device = "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed
wandb_path = "wandb:spd-resid-mlp/runs/svih4aik"
# Load the pretrained SPD model
model, config, label_coeffs = ResidualMLPSPDRankPenaltyModel.from_pretrained(wandb_path)
assert isinstance(config.task_config, ResidualMLPTaskConfig)
# Path must be local
target_model, target_train_config_dict, target_label_coeffs = ResidualMLPModel.from_pretrained(
    config.task_config.pretrained_model_path
)
assert torch.allclose(target_label_coeffs, torch.tensor(label_coeffs))
dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=False,  # Our labels will be the output of the target model
    data_generation_type=config.task_config.data_generation_type,
)
batch, labels = dataset.generate_batch(config.batch_size)
# Print some basic information about the model
print(f"Number of features: {model.config.n_features}")
print(f"Embedding dimension: {model.config.d_embed}")
print(f"MLP dimension: {model.config.d_mlp}")
print(f"Number of layers: {model.config.n_layers}")
print(f"Number of subnetworks (k): {model.config.k}")

target_model_output, _, _ = target_model(batch)

assert config.topk is not None
spd_outputs = run_spd_forward_pass(
    spd_model=model,
    target_model=target_model,
    input_array=batch,
    attribution_type=config.attribution_type,
    spd_type=config.spd_type,
    batch_topk=config.batch_topk,
    topk=config.topk,
    distil_from_target=config.distil_from_target,
)
# Topk recon (Note that we're using true labels not the target model output)
topk_recon_loss = calc_recon_mse(
    spd_outputs.spd_topk_model_output, target_model_output, has_instance_dim=True
)
print(f"Topk recon loss: {np.array(topk_recon_loss.detach())}")
# print(f"batch:\n{batch[:10]}")
# print(f"labels:\n{labels[:10]}")
# print(f"spd_outputs.spd_topk_model_output:\n{spd_outputs.spd_topk_model_output[:10]}")

# model.W_E @ target_model.layers[0].input_layer.weight.T
# in_matrix = einops.einsum(
#     model.W_E,
#     target_model.layers[0].input_layer.weight.T,
#     "n_instances n_features d_embed, n_instances d_embed n_features1 -> n_instances n_features n_features1",
# )

# Print param shapes for model
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# %% Do usual interp plots


class spd_dummy(ResidualMLPModel):
    def __init__(self):
        self.n_features = model.n_features
        self.n_instances = model.n_instances

    def __call__(self, batch: Float[Tensor, "batch n_instances"]):
        assert config.topk is not None
        return (
            run_spd_forward_pass(
                spd_model=model,
                target_model=target_model,
                input_array=batch,
                attribution_type=config.attribution_type,
                spd_type=config.spd_type,
                batch_topk=config.batch_topk,
                topk=config.topk,
                distil_from_target=config.distil_from_target,
            ).spd_topk_model_output,
            None,
            None,
        )


plot_individual_feature_response(spd_dummy(), device, target_train_config_dict)
plot_individual_feature_response(spd_dummy(), device, target_train_config_dict, sweep=True)


# %%

fig = plt.figure(constrained_layout=True, figsize=(10, 50))
gs = fig.add_gridspec(ncols=2, nrows=20 + 1 + 2)
ax_ID = fig.add_subplot(gs[:2, :])
ax1 = fig.add_subplot(gs[2, 0])
ax2 = fig.add_subplot(gs[2, 1])
virtual_weights = spd_calculate_virtual_weights(model, device, k_select="sum")
plot_virtual_weights(virtual_weights, device, ax1=ax1, ax2=ax2, ax3=ax_ID)
ax1.set_ylabel("sum over k")

norm = Normalize(vmin=-1, vmax=1)
for ki in range(model.k):
    ax1 = fig.add_subplot(gs[3 + ki, 0])
    ax2 = fig.add_subplot(gs[3 + ki, 1])
    virtual_weights = spd_calculate_virtual_weights(model, device, k_select=ki)
    plot_virtual_weights(virtual_weights, device, ax1=ax1, ax2=ax2, norm=norm)
    ax1.set_ylabel(f"k={ki}")
plt.show()
# %%
fig, axes1 = plt.subplots(21, 1, figsize=(10, 30), constrained_layout=True)
axes1 = np.atleast_1d(axes1)  # type: ignore
fig, axes2 = plt.subplots(21, 1, figsize=(5, 30), constrained_layout=True)
axes2 = np.atleast_1d(axes2)  # type: ignore
virtual_weights = spd_calculate_virtual_weights(model, device, k_select="sum")
relu_contribution_plot(axes1[0], axes2[0], virtual_weights, model, device)
for k in range(model.k):
    virtual_weights = spd_calculate_virtual_weights(model, device, k_select=k)
    relu_contribution_plot(axes1[k + 1], axes2[k + 1], virtual_weights, model, device)
    axes1[k + 1].set_ylabel(f"k={k}")
    axes2[k + 1].set_ylabel(f"k={k}")
    if k < model.k - 1:
        axes1[k + 1].set_xlabel("")
        axes2[k + 1].set_xlabel("")
plt.show()

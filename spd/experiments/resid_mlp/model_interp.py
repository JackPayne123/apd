# %% Imports
import matplotlib.pyplot as plt
import torch

from spd.experiments.resid_mlp.models import ResidualMLPModel
from spd.experiments.resid_mlp.plotting import (
    calculate_virtual_weights,
    plot_2d_snr,
    plot_individual_feature_response,
    plot_virtual_weights,
    relu_contribution_plot,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.utils import REPO_ROOT, set_seed

# %% Load model and config
model = "resid_mlp_identity_act_plus_resid_n-instances8_n-features20_d-resid200_d-mlp10_n-layers1_seed0_p0.05_random_embedding_True_identity_embedding_False_bias_False_False"
path = REPO_ROOT / "spd/experiments/resid_mlp/out" / model / "target_model.pth"

set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, task_config, label_coeffs = ResidualMLPModel.from_pretrained(path)
model = model.to(device)
dataset = ResidualMLPDataset(
    n_instances=task_config["n_instances"],
    n_features=task_config["n_features"],
    feature_probability=task_config["feature_probability"],
    device=device,
    label_type=task_config["label_type"],
    act_fn_name=task_config["act_fn_name"],
    label_fn_seed=task_config["label_fn_seed"],
    label_coeffs=label_coeffs,
    data_generation_type=task_config["data_generation_type"],
)
batch, labels = dataset.generate_batch(task_config["batch_size"])

# %% Plot feature response with one active feature
fig = plot_individual_feature_response(
    model,
    device,
    task_config,
    sweep=False,
)
fig = plot_individual_feature_response(
    model,
    device,
    task_config,
    sweep=True,
)
plt.show()

# %% Show connection strength between ReLUs and features
virtual_weights = calculate_virtual_weights(model=model, device=device)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)  # type: ignore

relu_contribution_plot(
    ax1=ax1, ax2=ax2, virtual_weights=virtual_weights, model=model, device=device, instance_idx=0
)
plt.show()

# %% Calculate S/N ratio for 1 and 2 active features.
fig = plot_2d_snr(model, device)
plt.show()


# %% Plot virtual weights

fig = plt.figure(constrained_layout=True, figsize=(20, 20))
gs = fig.add_gridspec(ncols=2, nrows=3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1:, :])
virtual_weights = calculate_virtual_weights(model=model, device=device)
fig = plot_virtual_weights(
    virtual_weights=virtual_weights, device=device, ax1=ax1, ax2=ax2, ax3=ax3, instance_idx=0
)
plt.show()

# %%

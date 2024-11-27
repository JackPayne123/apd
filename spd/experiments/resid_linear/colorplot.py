import einops
import matplotlib.pyplot as plt
import torch

from spd.experiments.resid_linear.models import ResidualLinearModel
from spd.experiments.resid_linear.resid_linear_dataset import ResidualLinearDataset
from spd.utils import set_seed

# %%

# Set up device and seed
device = "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed

# Load model and config
path = "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/resid_linear/out/resid_linear_identity_n-features100_d-resid1000_d-mlp50_n-layers1_seed0/target_model.pth"
model, task_config, label_coeffs = ResidualLinearModel.from_pretrained(path)
print(task_config)
model = model.to(device)
task_config["batch_size"] = 128
dataset = ResidualLinearDataset(
    embed_matrix=model.W_E,
    n_features=model.n_features,
    feature_probability=task_config["feature_probability"],
    device=device,
    label_coeffs=label_coeffs,
    data_generation_type="exactly_one_active",
)
batch, labels = dataset.generate_batch(task_config["batch_size"])
# %%

batch, labels = dataset.generate_batch(task_config["batch_size"])
for f in torch.arange(0, 100, 1):
    batch = torch.zeros_like(batch)
    batch[:, f] = -1 + 2 * 1  # torch.linspace(0, 1, 100)
    # batch[:, 1::2] = -1 + 2 * 0.25
    out, pre_acts, _ = model(batch)
    embed = pre_acts["layers.0.input_layer.weight"]
    mlp_out = out - embed
    feature_out = einops.einsum(
        mlp_out, model.W_E, "batch d_embed, n_features d_embed  -> batch n_features"
    )
    cmap_viridis = plt.get_cmap("viridis")
    color = cmap_viridis(f / model.n_features)
    plt.plot(
        feature_out[0, :].detach().cpu().numpy(),
        #  / [0, f].detach().cpu().numpy(),
        color=color,
    )
    # labels
    labels = torch.relu(batch)
    # plt.scatter(
    #     torch.arange(model.n_features),
    #     labels[0, :].detach().cpu().numpy(),
    #     color=color,
    # )
    # plt.ylim(-0.21, 1.21)
    plt.title(f"p={task_config['feature_probability']}")
    # plt.show()

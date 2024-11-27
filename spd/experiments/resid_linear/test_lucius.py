# %%
import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from torch import Tensor
from tqdm import tqdm

from spd.experiments.resid_linear.models import ResidualLinearModel
from spd.experiments.resid_linear.resid_linear_dataset import ResidualLinearDataset
from spd.utils import set_seed

# %%

# Set up device and seed
device = "cpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed

# Load model and config
path = "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/resid_linear/out/resid_linear_identity_n-features100_d-resid1000_d-mlp50_n-layers1_seed0/target_model.pth"

model, task_config, label_coeffs = ResidualLinearModel.from_pretrained(path)
model = model.to(device)


dataset = ResidualLinearDataset(
    embed_matrix=model.W_E,
    n_features=model.n_features,
    feature_probability=task_config["feature_probability"],
    device=device,
    label_coeffs=label_coeffs,
    data_generation_type="exactly_one_active",
)
task_config["batch_size"] = 128
batch, resid_labels = dataset.generate_batch(task_config["batch_size"])
labels = batch + F.relu(batch)

plt.title("W_E @ W_E^T")
plt.imshow((model.W_E @ model.W_E.T).detach(), cmap="RdBu", norm=CenteredNorm())
plt.colorbar()

# %%

for p in 2 ** torch.arange(1) / 30000:
    model.W_E.data = torch.eye(model.W_E.data.shape[0])
    model.layers[0].input_layer.weight.data.fill_(0)
    model.layers[0].input_layer.weight.data[
        torch.rand(model.layers[0].input_layer.weight.data.shape) < p
    ] = 1
    model.layers[0].output_layer.weight.data[:, :] = model.layers[0].input_layer.weight.data.T
    model.layers[0].output_layer.weight.data = model.layers[0].output_layer.weight.data / (
        1e-16 + model.layers[0].output_layer.weight.data.norm(dim=1, keepdim=True) ** 2
    )

    out, pre, post = model(batch)
    out = out.cpu().detach()
    print(f"p={p}, MSE loss", F.mse_loss(out, labels) * out.shape[-1])

print(1 / 4 * 1 / 3 * 1 / 4)

# %%
indices = batch.abs().argmax(dim=1)
plt.scatter(batch[torch.arange(batch.shape[0]), indices], out[torch.arange(out.shape[0]), indices])
# 0th entry
print(batch[0, indices[0]])
print(out[0, indices[0]])
print(out[0, :])
print(labels[0, indices[0]])
# 1st entry
print(batch[1, indices[1]])
print(out[1, indices[1]])
print(labels[1, indices[1]])
# 2nd entry
print(batch[2, indices[2]])
print(out[2, indices[2]])
print(labels[2, indices[2]])

# %%
plt.plot(out[1, :])
# %%
batch, resid_labels = dataset.generate_batch(task_config["batch_size"])

out, pre, post = model(batch)
idx = out[1, :].argmax(dim=0)
print(idx)
print(model.layers[0].input_layer.weight.data.shape)
print(torch.where(model.layers[0].input_layer.weight.data[:, idx] > 0))
# %%

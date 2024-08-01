# %%
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import plotly.express as px
import torch
from jaxtyping import Float
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader

from spd.log import logger
from spd.models.bool_circuit_models import BoolCircuitTransformer
from spd.scripts.bool_circuits.bool_circuit_utils import (
    create_circuit_str,
    create_truth_table,
    make_detailed_circuit,
    plot_circuit,
)
from spd.scripts.bool_circuits.train_bool_circuit import (
    Config,
    evaluate_model,
    get_circuit,
    get_train_test_dataloaders,
)
from spd.types import RootPath

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Load model, config, circuit, truth table, dataloaders, and evaluate model

out_dir: RootPath = Path(__file__).parent / "out/inp10-op20-hid8-lay1-circseed1-seed0"

with open(out_dir / "config.json") as f:
    config = Config(**json.load(f))
logger.info(f"Config loaded from {out_dir / 'config.json'}")

trained_model = BoolCircuitTransformer(
    n_inputs=config.n_inputs,
    d_embed=config.d_embed,
    d_mlp=config.d_embed,
    n_layers=config.n_layers,
).to(device)
trained_model.load_state_dict(torch.load(out_dir / "model.pt"))
logger.info(f"Model loaded from {out_dir / 'model.pt'}")

circuit = get_circuit(config)
circuit = make_detailed_circuit(circuit, config.n_inputs)
plot_circuit(circuit, config.n_inputs, show_out_idx=True)
logger.info(f"Circuit: n_inputs={config.n_inputs} - {circuit}")
logger.info(f"Circuit string: {create_circuit_str(circuit, config.n_inputs)}")

handcoded_model = BoolCircuitTransformer(
    n_inputs=config.n_inputs,
    d_embed=30,  # exact minimum for my handcode of that circuit
    d_mlp=7,  # exact minimum for my handcode of that circuit
    n_layers=7,  # exact minimum for my handcode of that circuit
).to(device)
handcoded_model.init_handcoded(circuit)
logger.info("Handcoded model initialized with circuit")


truth_table = create_truth_table(config.n_inputs, circuit)
logger.info(f"Truth table:\n{truth_table}")

train_dataloader, eval_dataloader = get_train_test_dataloaders(config, circuit)
eval_loss = evaluate_model(trained_model, eval_dataloader, device)
logger.info(f"Evaluation loss: {eval_loss}")
assert eval_loss < 1e-5, f"Unusual evaluation loss of {eval_loss:.3e}"

eval_loss_handcoded = evaluate_model(
    handcoded_model, eval_dataloader, device, output_is_logit=False
)
logger.info(f"Handcoded model evaluation loss: {eval_loss_handcoded}")


# %%

for inputs, labels in train_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    print(f"{inputs.shape=}, {inputs}")
    preds = trained_model(inputs)
    probabilities = torch.sigmoid(preds)
    print(f"{preds.shape=}, {preds:}")
    print(f"{probabilities.shape=}, {probabilities}")
    print(f"{labels.shape=}, {labels}")
    break

# %% Collect activations via PyTorch hooks


def cache_batch_activations(
    model: BoolCircuitTransformer, inputs: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    activations = {}

    def mlp_hook_fn(
        name: str,
    ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def fn(
            module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            activations[f"{name}_in"] = inputs[0].detach()
            activations[f"{name}_mid"] = module.linear1(inputs[0]).detach()
            activations[f"{name}_out"] = output.detach()

        return fn

    handles = []
    for i, layer in enumerate(model.layers):
        hook_fn = mlp_hook_fn(f"mlp_{i}")
        handles.append(layer.register_forward_hook(hook_fn))

    outputs = model(inputs)

    for handle in handles:
        handle.remove()

    return activations, outputs


def cache_all_activations(
    model: BoolCircuitTransformer,
    dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    device: Literal["cpu", "cuda"],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    all_activations = {}
    all_inputs = []
    all_outputs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            activations, outputs = cache_batch_activations(model, inputs)

            all_outputs.append(outputs.cpu())

            for name, activation in activations.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(activation.cpu())

            all_inputs.append(inputs.cpu())
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    for name in all_activations:
        all_activations[name] = torch.cat(all_activations[name], dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_activations, {
        "inputs": all_inputs,
        "labels": all_labels,
        "outputs": all_outputs,
    }


# %%

# %% Run collection of activations for both trained and handcoded models
trained_cache_acts, trained_cache = cache_all_activations(trained_model, train_dataloader, device)
handcoded_cache_acts, handcoded_cache = cache_all_activations(
    handcoded_model, train_dataloader, device
)

# %% Handcoded model
print(f"Inputs shape: {handcoded_cache['inputs'].shape}")
print("Shapes of cached data for trained model:")
for name, activation in handcoded_cache_acts.items():
    print(f"{name}: {activation.shape}")
print(f"Labels shape: {handcoded_cache['labels'].shape}")

# Print all out activations of the first batch
print(f"{'Input':25} = {' '.join([str(x.int().item()) for x in handcoded_cache['inputs'][0]])}")
for name, activation in handcoded_cache_acts.items():
    print(f"{name:25} = {' '.join([str(x.int().item()) for x in activation[0]])}")
print(f"{'Label':25} = {handcoded_cache['labels'][0].int().item()}")

# %% We can read everything off by imshowing the matrices... skipping to trained model.
print("Shapes of cached data for trained model:")
for name, activation in trained_cache_acts.items():
    print(f"{name}: {activation.shape}")
print(f"Inputs shape: {trained_cache['inputs'].shape}")
print(f"Labels shape: {trained_cache['labels'].shape}")

# %% Step 1: Which inputs matter. For each input randomize it and see if it makes a difference
orig_outputs = trained_cache["outputs"]
orig_probs = torch.sigmoid(orig_outputs)
n_inputs = config.n_inputs
for in_idx in range(n_inputs):
    inputs = torch.tensor(trained_cache["inputs"])
    inputs[:, in_idx] = torch.randint(0, 2, (len(inputs),))
    new_outputs = trained_model(inputs.to(device))
    new_probs = torch.sigmoid(new_outputs)
    diff = torch.abs(new_probs - orig_probs)
    print(f"{in_idx}", f"Mean diff: {diff.mean().item():.3f}, Max diff: {diff.max().item():.3f}")


# %% Step 2: PCA coloured by the inputs we know matter
# We know input 1, 2, 3, and 8 matter. Let's make input 1 = x or o, and the other 3 in 8 colours.
inputs = torch.tensor(trained_cache["inputs"])
color_number = 1 * inputs[:, 1] + 2 * inputs[:, 3] + 4 * inputs[:, 8]
hover = [
    f"{x[2].int().item()}{x[2].int().item()}{x[3].int().item()}{x[8].int().item()}" for x in inputs
]

# PCA embeddings aka mlp_0_in
pca = PCA(n_components=8)
pca_act = torch.tensor(pca.fit_transform(trained_cache_acts["mlp_0_in"]))
print("Variance explained by each component:", [f"{x:.2%}" for x in pca.explained_variance_ratio_])

tab10 = plt.get_cmap("tab10")(range(10))
tab10_colors = [tab10[int(x)] for x in color_number]
px_colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b, _ in tab10_colors]
fig = px.scatter_3d(
    x=pca_act[:, 0],
    y=pca_act[:, 1],
    z=pca_act[:, 2],
    color=px_colors,
    symbol=inputs[:, 1],
    opacity=0.7,
    hover_data={"color": color_number, "hover": hover},
)
fig.show()

# %%

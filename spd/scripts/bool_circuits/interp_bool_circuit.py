# %%
import json
from pathlib import Path

import torch

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


def get_activations(model: BoolCircuitTransformer, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    activations = {}

    def hook_fn(name: str):
        def fn(_, __, output):
            activations[name] = output.detach()

        return fn

    handles = []
    handles.append(model.W_E.register_forward_hook(hook_fn("W_E")))
    handles.append(model.W_U.register_forward_hook(hook_fn("W_U")))

    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(hook_fn(f"layer_{i}")))

    model(inputs)

    for handle in handles:
        handle.remove()

    return activations


# Function to cache activations
def cache_activations(
    model: BoolCircuitTransformer, dataloader: torch.utils.data.DataLoader, device: str
) -> dict[str, torch.Tensor]:
    all_activations = {}
    all_inputs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            activations = get_activations(model, inputs)

            for name, activation in activations.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(activation.cpu())

            all_inputs.append(inputs.cpu())
            all_labels.append(labels)

    # Concatenate all activations, inputs, and labels
    for name in all_activations:
        all_activations[name] = torch.cat(all_activations[name], dim=0)

    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {"activations": all_activations, "inputs": all_inputs, "labels": all_labels}


# Cache activations for both trained and handcoded models
trained_cache = cache_activations(trained_model, train_dataloader, device)
handcoded_cache = cache_activations(handcoded_model, train_dataloader, device)

logger.info("Activations cached for both trained and handcoded models")
# %% Example of how to access the cached data
print("Shapes of cached data for trained model:")
for name, activation in handcoded_cache["activations"].items():
    print(f"{name}: {activation.shape}")
print(f"Inputs shape: {handcoded_cache['inputs'].shape}")
print(f"Labels shape: {handcoded_cache['labels'].shape}")

# %%

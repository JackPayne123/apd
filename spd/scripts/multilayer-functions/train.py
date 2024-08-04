# %%
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from multifunction_play import generate_regular_simplex, generate_trig_functions
from piecewise_linear import ControlledResNet, FunctionModule

# %%

num_functions = 50
dim = 50

functions = generate_trig_functions(num_functions)

if num_functions == dim:
    control_W_E = torch.eye(num_functions)
elif num_functions == dim + 1:
    control_W_E = generate_regular_simplex(num_functions)
    control_W_E = control_W_E / control_W_E.norm(dim=1).unsqueeze(1)
else:
    control_W_E = torch.randn(num_functions, dim)
    control_W_E = control_W_E / control_W_E.norm(dim=1).unsqueeze(1)


@dataclass
class Config:
    num_functions: int = 50
    d_control: int = 40
    lr: float = 1e-3
    num_epochs: int = 100
    num_layers: int = 5
    num_neurons: int = 40
    start: float = 0
    end: float = 5
    batch_size: int = 32
    control_prob: float = 1 / 50
    control_prob_decay: float = 0.9
    plot_freq: int = 10


def generate_data(config: Config) -> torch.Tensor:
    inputs = config.start + torch.rand(config.batch_size, 1) * (config.end - config.start)

    control_probs = config.control_prob * (
        config.control_prob_decay ** torch.arange(config.num_functions)
    )

    control_bits = torch.rand(config.batch_size, config.num_functions) < control_probs.unsqueeze(0)

    # each datapoint is an input followed by a set of control bits
    dataset = torch.cat([inputs, control_bits], dim=1)
    return dataset


def train(config: Config) -> nn.Module:
    functions = generate_trig_functions(config.num_functions)
    function_module = FunctionModule(functions)
    print("functions generated")
    # target = ControlledResNet(
    #     function_module,
    #     config.start,
    #     config.end,
    #     config.num_neurons,
    #     config.num_layers,
    #     config.num_functions,
    #     negative_suppression=1000,
    # )
    # # no gradients to target
    # for param in target.parameters():
    #     param.requires_grad = False

    network = ControlledResNet(
        function_module,
        config.start,
        config.end,
        config.num_neurons,
        config.num_layers,
        config.d_control,
        random_params=True,
    )
    print("networks initialised")

    opt = torch.optim.Adam(network.parameters(), lr=config.lr)

    for _ in range(config.num_epochs):
        data = generate_data(config)
        opt.zero_grad()
        target_output = network.ideal_forward(data)
        network_output = network(data)[:, 0]
        loss = F.mse_loss(network_output, target_output)
        loss.backward()
        opt.step()
        # print some diagnostics
        print(f"loss: {loss.item()}")

    return network


target_network = ControlledResNet(functions, 0, 5, 40, 5, dim, negative_suppression=100)

initialised_network = ControlledResNet(
    functions, 0, 5, 40, 5, dim, negative_suppression=100, random_params=True
)

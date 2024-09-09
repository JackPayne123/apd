import json
from collections.abc import Callable
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import collections as mc
from matplotlib import colors as mcolors
from pydantic import BaseModel, PositiveInt
from tqdm import tqdm, trange

from spd.experiments.piecewise.models import PiecewiseFunctionTransformer
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.trig_functions import generate_trig_functions
from spd.utils import BatchedDataLoader, set_seed


class PiecewiseTrainConfig(BaseModel):
    batch_size: PositiveInt
    steps: PositiveInt
    n_functions: PositiveInt = 10
    neurons_per_function: PositiveInt = 10
    n_layers: PositiveInt = 1
    feature_probability: float = 0.05
    range_min: float = 0.0
    range_max: float = 5.0
    seed: int = 0


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def train(
    model: PiecewiseFunctionTransformer,
    dataloader: BatchedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    steps: int = 5_000,
    print_freq: int = 100,
    lr: float = 5e-3,
    lr_schedule: Callable[[int, int], float] = linear_lr,
) -> None:
    hooks = []

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch, labels = next(data_iter)
            out = model(batch)
            error = importance * (labels.abs() - out) ** 2
            loss = einops.reduce(error, "b i f -> i", "mean").sum()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(
                    model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr
                )
                for h in hooks:
                    h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item() / model.n_instances}")
                t.set_postfix(
                    loss=loss.item() / model.n_instances,
                    lr=step_lr,
                )


def plot_intro_diagram(model: TMSModel, filepath: Path) -> None:
    WA = model.W.detach()
    N = len(WA[:, 0])
    sel = range(config.n_instances)  # can be used to highlight specific sparsity levels
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color",
        # plt.cm.viridis(model.importance[0].cpu().numpy()),  # type: ignore
        plt.cm.viridis(np.array([1.0])),  # type: ignore
    )
    plt.rcParams["figure.dpi"] = 200
    fig, axs = plt.subplots(1, len(sel), figsize=(2 * len(sel), 2))
    axs = np.array(axs)
    for i, ax in zip(sel, axs, strict=False):
        W = WA[i].cpu().detach().numpy()
        colors = [mcolors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
        ax.scatter(W[:, 0], W[:, 1], c=colors[0 : len(W[:, 0])])
        ax.set_aspect("equal")
        ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=colors))  # type: ignore

        z = 1.5
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z, z))
        ax.set_ylim((-z, z))
        ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position("center")
    plt.savefig(filepath)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PiecewiseTrainConfig(
        batch_size=1024,
        steps=5_000,
        n_functions=10,
        neurons_per_function=10,
        n_layers=4,
        feature_probability=0.05,
        range_min=0.0,
        range_max=5.0,
        seed=0,
    )

    set_seed(config.seed)

    assert (
        config.neurons_per_function * config.n_functions % config.n_layers == 0
    ), "neurons per function times num functions must be divisible by num layers."
    d_mlp = config.neurons_per_function * config.n_functions // config.n_layers
    model = PiecewiseFunctionTransformer(
        n_inputs=config.n_functions + 1,
        d_mlp=d_mlp,
        n_layers=config.n_layers,
        handcoded=False,
    )
    functions, function_params = generate_trig_functions(config.n_functions)

    dataset = PiecewiseDataset(
        n_inputs=model.n_inputs,
        functions=functions,
        feature_probability=config.feature_probability,
        range_min=config.range_min,
        range_max=config.range_max,
        batch_size=config.batch_size,
        return_labels=False,
    )
    dataloader = BatchedDataLoader(dataset)

    train(model, dataloader=dataloader, steps=config.steps)

    run_name = (
        f"piecewise_n-functions{config.n_functions}_neur-per-fun{config.neurons_per_function}_"
        f"n-layers{config.n_layers}_seed{config.seed}_feature-prob{config.feature_probability}.pth"
    )
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pth")
    print(f"Saved model to {out_dir / 'model.pth'}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=4)
    print(f"Saved config to {out_dir / 'config.json'}")

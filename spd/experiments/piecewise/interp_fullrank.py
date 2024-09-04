# %%
import json
from pathlib import Path

import torch

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_decomposition import plot_components_fullrank
from spd.experiments.piecewise.trig_functions import create_trig_function
from spd.run_spd import Config, PiecewiseConfig

pretrained_path = Path(
    "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/piecewise/out/testplot_lay1_lr0.001_p0.9_topk0.625_topkrecon1.0_lpspNone_topkl2_0.01_bs2048/model_50000.pth"
)
with open(pretrained_path.parent / "config.json") as f:
    config = Config(**json.load(f))

with open(pretrained_path.parent / "function_params.json") as f:
    function_params = json.load(f)
functions = [create_trig_function(*param) for param in function_params]

device = "cuda" if torch.cuda.is_available() else "cpu"

assert isinstance(config.task_config, PiecewiseConfig)
hardcoded_model = PiecewiseFunctionTransformer.from_handcoded(
    functions=functions,
    neurons_per_function=config.task_config.neurons_per_function,
    n_layers=config.task_config.n_layers,
    range_min=config.task_config.range_min,
    range_max=config.task_config.range_max,
    seed=config.seed,
).to(device)
hardcoded_model.eval()

model = PiecewiseFunctionSPDFullRankTransformer(
    n_inputs=hardcoded_model.n_inputs,
    d_mlp=hardcoded_model.d_mlp,
    n_layers=hardcoded_model.n_layers,
    k=config.task_config.k,
)
model.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))
model.to(device)

# %%
topk = config.topk
batch_topk = config.batch_topk
fig = plot_components_fullrank(model=model, step=-1, out_dir=None, device=device, slow_images=True)
fig.savefig("out/fullrank_components.png", dpi=300)

# %%

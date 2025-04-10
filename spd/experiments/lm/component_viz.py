"""
Vizualises the components of the model.
"""

from pathlib import Path

import torch
import wandb
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from wandb.apis.public import Run

from spd.configs import Config, LMTaskConfig
from spd.experiments.lm.models import SSModel
from spd.types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


def load_model(
    path: ModelPath,
) -> tuple[SSModel, Config]:
    if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
        wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
        api = wandb.Api()
        run: Run = api.run(wandb_path)
        checkpoint_file_obj = fetch_latest_wandb_checkpoint(run, prefix="model")

        run_dir = fetch_wandb_run_dir(run.id)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint_file_obj.name)

    else:
        checkpoint_path = Path(path)  # local path

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    config = Config(**checkpoint_dict["config"])

    assert isinstance(config.task_config, LMTaskConfig)
    model_config_dict = MODEL_CONFIGS[config.task_config.model_size]
    model_path = f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}"
    llama_model = Llama.from_pretrained(model_path, model_config_dict)

    ss_model = SSModel(
        llama_model=llama_model,
        target_module_patterns=config.task_config.target_module_patterns,
        m=config.m,
        n_gate_hidden_neurons=config.n_gate_hidden_neurons,
    )
    ss_model.load_state_dict(checkpoint_dict["model"])
    return ss_model, config


def main(path: ModelPath) -> None:
    ss_model, config = load_model(path)
    print(ss_model)
    print(config)


if __name__ == "__main__":
    path = "wandb:spd-lm/runs/60ycavou"
    main(path)

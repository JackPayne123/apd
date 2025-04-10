"""
Vizualises the components of the model.
"""

from spd.experiments.lm.models import SSModel
from spd.types import ModelPath


def main(path: ModelPath) -> None:
    ss_model, config = SSModel.from_pretrained(path)
    print(ss_model)
    print(config)


if __name__ == "__main__":
    path = "wandb:spd-lm/runs/60ycavou"
    main(path)

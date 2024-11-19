"""Trains a residual linear model on one-hot input vectors."""

import json
from pathlib import Path
from typing import Literal, Self

import einops
import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn
from torch.nn import functional as F

from spd.experiments.resid_linear.models import ResidualLinearModel
from spd.experiments.resid_linear.resid_linear_dataset import (
    ResidualLinearDataset,
)
from spd.run_spd import get_lr_schedule_fn
from spd.utils import DatasetGeneratedDataLoader, set_seed

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    label_fn_seed: int = 0
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    feature_probability: PositiveFloat
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    random_embedding_matrix: bool = False
    act_fn_name: Literal["gelu", "relu"] = "gelu"

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not self.random_embedding_matrix:
            assert self.n_features == self.d_embed, (
                "n_features must equal d_embed if we are not using a random "
                "embedding matrix, since this will use an identity matrix"
            )
        return self


def train(
    config: Config,
    model: ResidualLinearModel,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_resid"]]
    ],
    device: str,
    out_dir: Path | None = None,
) -> float | None:
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    # Add this line to get the lr_schedule_fn
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)

    final_loss = None
    for step, (batch, labels) in enumerate(dataloader):
        if step >= config.steps:
            break

        # Add this block to update the learning rate
        current_lr = config.lr * lr_schedule_fn(step, config.steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)
        out, _, _ = model(batch)
        loss = F.mse_loss(out, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % config.print_freq == 0:
            print(f"Step {step}: loss={final_loss}, lr={current_lr}")

    if out_dir is not None:
        model_path = out_dir / "target_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        config_path = out_dir / "target_model_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, indent=2)
        print(f"Saved config to {config_path}")

        # Save the coefficients used to generate the labels
        assert isinstance(dataloader.dataset, ResidualLinearDataset)
        label_coeffs = dataloader.dataset.coeffs.tolist()
        label_coeffs_path = out_dir / "label_coeffs.json"
        with open(label_coeffs_path, "w") as f:
            json.dump(label_coeffs, f)
        print(f"Saved label coefficients to {label_coeffs_path}")

    print(f"Final loss: {final_loss}")
    return final_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        seed=0,
        label_fn_seed=0,
        n_features=40,
        d_embed=20,
        d_mlp=40,
        n_layers=1,
        feature_probability=0.05,
        batch_size=1024,
        steps=40_000,
        print_freq=100,
        lr=3e-3,
        lr_schedule="cosine",
        random_embedding_matrix=True,
        act_fn_name="gelu",
    )

    set_seed(config.seed)
    run_name = (
        f"resid_linear_identity_n-features{config.n_features}_d-resid{config.d_embed}_"
        f"d-mlp{config.d_mlp}_n-layers{config.n_layers}_seed{config.seed}"
    )
    out_dir = Path(__file__).parent / "out" / run_name

    model = ResidualLinearModel(
        n_features=config.n_features,
        d_embed=config.d_embed,
        d_mlp=config.d_mlp,
        n_layers=config.n_layers,
        act_fn_name=config.act_fn_name,
    ).to(device)

    # Init embedding
    assert model.W_E.shape == (config.n_features, config.d_embed)
    if config.random_embedding_matrix:
        model.W_E.data[:, :] = torch.randn(config.n_features, config.d_embed, device=device)
        # Ensure they are norm 1
        model.W_E.data /= model.W_E.data.norm(dim=1, keepdim=True)
    else:
        # Make W_E the identity matrix
        model.W_E.data[:, :] = torch.eye(config.d_embed, device=device)

    # Don't train the Embedding matrix
    model.W_E.requires_grad = False
    trainable_params = [p for n, p in model.named_parameters() if "W_E" not in n]

    out_dir.mkdir(parents=True, exist_ok=True)
    # Plot the cosine similarities of each n_features rows of the embedding matrix.
    # Plot a histogram of the angles between each pair of rows.
    if config.random_embedding_matrix:
        cosine_similarities = einops.einsum(
            model.W_E,
            model.W_E,
            "n_features_0 d_embed, n_features_1 d_embed -> n_features_0 n_features_1",
        )
        # Zero out the diagonal
        cosine_similarities[torch.arange(config.n_features), torch.arange(config.n_features)] = 0
        plt.hist(cosine_similarities.flatten().tolist())
        plt.savefig(out_dir / "cosine_similarities.png")
        print(f"Saved cosine similarities to {out_dir / 'cosine_similarities.png'}")
        plt.close()

    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        label_fn_seed=config.label_fn_seed,
        act_fn_name=config.act_fn_name,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    train(
        config=config,
        model=model,
        trainable_params=trainable_params,
        dataloader=dataloader,
        device=device,
        out_dir=out_dir,
    )

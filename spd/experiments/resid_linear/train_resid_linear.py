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
from tqdm import tqdm

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
    loss_at_resid: bool = True
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
    data_generation_type: Literal[
        "at_least_zero_active", "exactly_one_active", "exactly_two_active", "exactly_three_active"
    ] = "at_least_zero_active"

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not self.random_embedding_matrix:
            assert self.n_features == self.d_embed, (
                "n_features must equal d_embed if we are not using a random "
                "embedding matrix, since this will use an identity matrix"
            )
        return self


def evaluate(
    model: ResidualLinearModel,
    eval_dataloaders: list[
        DatasetGeneratedDataLoader[
            tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_resid"]]
        ]
    ],
) -> tuple[list[float], list[float]]:
    def calc_losses(model, batch, labels):
        W_E = model.W_E.detach()
        labels_resid = labels
        labels_features = batch + torch.relu(batch)
        out_resid, _, _ = model(batch)
        out_features = einops.einsum(
            W_E, out_resid, "n_features d_embed, batch d_embed -> batch n_features"
        )
        eval_loss_resid = F.mse_loss(out_resid, labels_resid) * out_resid.shape[-1]
        eval_loss_features = F.mse_loss(out_features, labels_features) * out_features.shape[-1]
        return eval_loss_resid.item(), eval_loss_features.item()

    def calc_trivial_losses(W_E, batch, labels):
        "Loss you would get if the model was just an identity"
        labels_resid = labels
        labels_features = batch + torch.relu(batch)
        out_features = einops.einsum(
            batch,
            W_E,
            W_E,
            "batch n_features, n_features d_embed, n_features_out d_embed -> batch n_features_out",
        )
        out_resid = einops.einsum(
            W_E,
            batch,
            "n_features d_embed, batch n_features -> batch d_embed",
        )
        eval_loss_resid = F.mse_loss(out_resid, labels_resid) * out_resid.shape[-1]
        eval_loss_features = F.mse_loss(out_features, labels_features) * out_features.shape[-1]
        return eval_loss_resid.item(), eval_loss_features.item()

    eval_losses = []
    id_losses = []
    for eval_dataloader in eval_dataloaders:
        batch, labels = next(iter(eval_dataloader))
        eval_losses.append(calc_losses(model, batch, labels))
        id_losses.append(calc_trivial_losses(model.W_E, batch, labels))
    return eval_losses, id_losses


def train(
    config: Config,
    model: ResidualLinearModel,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_resid"]]
    ],
    eval_dataloaders: list[
        DatasetGeneratedDataLoader[
            tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_resid"]]
        ]
    ],
    device: str,
    out_dir: Path | None = None,
) -> float | None:
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    # Add this line to get the lr_schedule_fn
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)

    final_loss = None
    for step, (batch, labels) in tqdm(enumerate(dataloader), total=config.steps):
        if step >= config.steps:
            break

        # Add this block to update the learning rate
        current_lr = config.lr * lr_schedule_fn(step, config.steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)
        labels = labels.detach()
        out, _, _ = model(batch)
        if config.loss_at_resid:
            loss = F.mse_loss(out, labels) * out.shape[-1]  # correct for mean in mse_loss
        else:
            labels = batch + torch.relu(batch)
            out = einops.einsum(
                model.W_E, out, "n_features d_embed, batch d_embed -> batch n_features"
            )
            loss = F.mse_loss(out, labels) * out.shape[-1]  # correct for mean in mse_loss

        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % config.print_freq == 0:
            print(f"Step {step}: loss={final_loss}, lr={current_lr}")
            # with torch.inference_mode():
            #     eval_losses, id_losses = evaluate(model, [dataloader, *eval_dataloaders])
            #     for eval_loss, id_loss in zip(eval_losses, id_losses, strict=False):
            #         print(f"Eval loss: {eval_loss}, id loss: {id_loss}")

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
        # print("Trying Lucius handcoded model:")
        # print("WE", model.W_E.shape, model.W_E)
        # # assert config.d_mlp == 3000
        # p = 20 / config.d_mlp
        # model.W_E.data = torch.eye(model.W_E.data.shape[0]).to(device)
        # model.layers[0].input_layer.weight.data.fill_(0)
        # model.layers[0].input_layer.weight.data[
        #     torch.rand(model.layers[0].input_layer.weight.data.shape) < p
        # ] = 1
        # model.layers[0].output_layer.weight.data[:, :] = model.layers[0].input_layer.weight.data.T
        # assert model.layers[0].output_layer.weight.data.shape == (config.d_embed, config.d_mlp)
        # model.layers[0].output_layer.weight.data = model.layers[0].output_layer.weight.data / (
        #     1e-16 + model.layers[0].output_layer.weight.data.norm(dim=1, keepdim=True) ** 2
        # )
        # with torch.inference_mode():
        #     eval_losses, id_losses = evaluate(model, [dataloader, *eval_dataloaders])
        #     for eval_loss, id_loss in zip(eval_losses, id_losses, strict=False):
        #         print(f"Eval loss: {eval_loss}, id loss: {id_loss}")

    print(f"Final loss: {final_loss}")
    return final_loss


def training_run(config: Config, device: str) -> float:
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

    fixed_coeffs = [1.0] * config.n_features
    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        label_fn_seed=config.label_fn_seed,
        act_fn_name=config.act_fn_name,
        label_coeffs=fixed_coeffs,
        data_generation_type=config.data_generation_type,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Create a set of evaluation datasets with 1, 2 and 3 features active
    eval_datasets = [
        ResidualLinearDataset(
            embed_matrix=model.W_E,
            n_features=config.n_features,
            feature_probability=config.feature_probability,
            device=device,
            label_fn_seed=config.label_fn_seed,
            act_fn_name=config.act_fn_name,
            label_coeffs=fixed_coeffs,
            data_generation_type=f"exactly_{num}_active",
        )
        for num in ["one", "two", "three"]
    ]
    eval_dataloaders = [
        DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        for dataset in eval_datasets
    ]

    return train(
        config=config,
        model=model,
        trainable_params=trainable_params,
        dataloader=dataloader,
        eval_dataloaders=eval_dataloaders,
        device=device,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_steps = 1000
    n_features = 100
    d_embed = 100
    d_mlp = 50
    p = 0.01
    data_generation_type = "at_least_zero_active"
    config = Config(
        seed=0,
        label_fn_seed=0,
        loss_at_resid=True,
        n_features=n_features,
        d_embed=d_embed,
        d_mlp=d_mlp,
        n_layers=1,
        feature_probability=p,
        batch_size=2048,
        steps=n_steps,
        print_freq=100,
        lr=3e-3,
        lr_schedule="cosine",
        random_embedding_matrix=True,
        act_fn_name="relu",
        data_generation_type=data_generation_type,
    )
    training_run(config, device)


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     for n_steps in [1000, 10_000]:
#         losses = {}
#         for d_embed in [10, 100, 1000, 10_000, 100_000]:
#             n_features = 100
#             d_mlp = 50
#             p = 0.01
#             data_generation_type = "at_least_zero_active"
#             print(f"Training for {n_steps} steps with d_embed={d_embed}")
#             config = Config(
#                 seed=0,
#                 label_fn_seed=0,
#                 loss_at_resid=False,
#                 n_features=n_features,
#                 d_embed=d_embed,
#                 d_mlp=d_mlp,
#                 n_layers=1,
#                 feature_probability=p,
#                 batch_size=2048,
#                 steps=n_steps,
#                 print_freq=100,
#                 lr=3e-3,
#                 lr_schedule="cosine",
#                 random_embedding_matrix=True,
#                 act_fn_name="relu",
#                 data_generation_type=data_generation_type,
#             )
#             losses[d_embed] = training_run(config, device)
#         print(losses)
#         plt.loglog(list(losses.keys()), list(losses.values()), label=f"{n_steps} steps")
#     plt.axhline(
#         p * 1 / 2 * 1 / 3 * n_features * (1 - d_mlp / n_features),
#         ls="--",
#         color="k",
#         label="naive loss",
#     )
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.legend()
#     filename = f"loss_scaling_n-features{n_features}_d-mlp{d_mlp}_p{p}_loss_at_resid={config.loss_at_resid}.png"
#     plt.savefig(filename)
#     print(f"Saved losses to '{filename}'")
#     plt.close()

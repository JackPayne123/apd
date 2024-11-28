import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.experiments.resid_mlp.train_resid_mlp import Config, run_train
from spd.settings import REPO_ROOT
from spd.utils import set_seed


def test_train(
    batch_size: int,
    n_steps: int,
    d_embed: int,
    p: float,
    bias: bool,
    d_features: int,
    d_mlp: int,
    fixed_random_embedding: bool,
    fixed_identity_embedding: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        seed=0,
        label_fn_seed=0,
        label_type="act_plus_resid",
        use_trivial_label_coeffs=True,
        n_instances=1,
        n_features=d_features,
        d_embed=d_embed,
        d_mlp=d_mlp,
        n_layers=1,
        act_fn_name="relu",
        apply_output_act_fn=False,
        data_generation_type="at_least_zero_active",
        in_bias=bias,
        out_bias=bias,
        feature_probability=p,
        importance_val=None,
        batch_size=batch_size,
        steps=n_steps,
        print_freq=100,
        lr=3e-3,
        lr_schedule="cosine",
        fixed_random_embedding=fixed_random_embedding,
        fixed_identity_embedding=fixed_identity_embedding,
    )

    set_seed(config.seed)

    return run_train(config, device)


if __name__ == "__main__":
    out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out"
    os.makedirs(out_dir, exist_ok=True)
    batch_size = 2048
    d_features = 100
    d_mlp = 50
    p = 0.01
    d_embed = 1000
    # Scale d_embed
    losses = {}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), constrained_layout=True)
    for bias in [True, False]:
        for i, embed in enumerate(["trained", "random"]):
            fixed_random_embedding = embed == "random"
            fixed_identity_embedding = embed == "identity"
            for n_steps in [500, 1000, 2000, 5000, 10_000]:
                losses[n_steps] = {}
                for d_embed in np.geomspace(100, 100_000, 4):
                    d_embed = int(d_embed)
                    print(f"Testing {n_steps} steps, {d_embed} d_embed")
                    losses[n_steps][d_embed] = test_train(
                        batch_size=batch_size,
                        n_steps=n_steps,
                        d_embed=d_embed,
                        p=p,
                        bias=bias,
                        d_features=d_features,
                        d_mlp=d_mlp,
                        fixed_random_embedding=fixed_random_embedding,
                        fixed_identity_embedding=fixed_identity_embedding,
                    )
            # Save losses to json
            with open(
                out_dir
                / f"losses_scale_embed_{d_embed=}_{bias=}_{embed=}_{p=}_{d_features=}_{d_mlp=}.json",
                "w",
            ) as f:
                json.dump(losses, f)
            # Make plot
            ax = axes[int(bias), i]  # type: ignore
            naive_loss = (
                (d_features - d_mlp) * (8 - 3 * p) * p / 48
                if bias
                else (d_features - d_mlp) * p / 6
            )
            for n_steps in losses:
                ax.plot(
                    list(losses[n_steps].keys()),
                    list(losses[n_steps].values()),
                    label=f"{n_steps} steps",
                )
            ax.set_title(f"{d_features=}, {d_mlp=}, {p=}, {bias=}, W_E={embed}", fontsize=8)
            ax.axhline(naive_loss, color="k", linestyle="--", label=f"Naive loss {naive_loss:.2e}")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.legend(loc="upper right")
            ax.set_xlabel("d_embed")
            ax.set_ylabel("Loss")
    fig.suptitle("Loss scaling with training steps")
    fig.savefig(out_dir / "loss_scaling_resid_mlp_training_d_embed.png")
    plt.show()
    # Scale p
    losses = {}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), constrained_layout=True)
    ps = np.geomspace(0.01, 1, 4)
    for bias in [True, False]:
        for i, embed in enumerate(["trained", "random"]):
            fixed_random_embedding = embed == "random"
            fixed_identity_embedding = embed == "identity"
            print(f"Setting {bias=} and {embed=}")
            for n_steps in [500, 1000, 2000, 10000]:
                losses[n_steps] = {}
                for p in ps:
                    print(f"Testing {n_steps} steps, {p} p")
                    losses[n_steps][p] = test_train(
                        batch_size=batch_size,
                        n_steps=n_steps,
                        d_embed=d_embed,
                        p=p,
                        bias=bias,
                        d_features=d_features,
                        d_mlp=d_mlp,
                        fixed_random_embedding=fixed_random_embedding,
                        fixed_identity_embedding=fixed_identity_embedding,
                    )
            # Save losses to json
            with open(
                out_dir
                / f"losses_scale_p_{bias=}_{embed=}_{p=}_{d_embed=}_{d_features=}_{d_mlp=}.json",
                "w",
            ) as f:
                json.dump(losses, f)
            # Make plot
            ax = axes[int(bias), i]  # type: ignore
            naive_losses = [
                (d_features - d_mlp) * (8 - 3 * p) * p / 48
                if bias
                else (d_features - d_mlp) * p / 6
                for p in ps
            ]
            ax.plot(ps, naive_losses, color="k", linestyle="--", label="Naive loss")
            for n_steps in losses:
                ax.plot(
                    list(losses[n_steps].keys()),
                    list(losses[n_steps].values()),
                    label=f"{n_steps} steps",
                )
            ax.set_title(f"{d_features=}, {d_embed=}, {d_mlp=}, {bias=}, W_E={embed}", fontsize=8)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.legend(loc="upper left")
            ax.set_xlabel("p")
            ax.set_ylabel("Loss")
    fig.suptitle("Loss scaling with p")
    plt.show()

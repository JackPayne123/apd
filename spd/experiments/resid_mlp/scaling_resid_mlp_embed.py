import json
import os

import matplotlib.pyplot as plt

from spd.experiments.resid_mlp.scaling_resid_mlp_training import (
    naive_loss,
    plot_loss_curve,
    train_on_test_data,
)
from spd.settings import REPO_ROOT

if __name__ == "__main__":
    out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out"
    os.makedirs(out_dir, exist_ok=True)
    n_instances = 20
    n_features = 100
    d_mlp = 100
    d_embed = None
    n_steps = None
    # Scale d_embed
    p = 0.01
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(f"Loss scaling with d_embed. Using {n_instances} instances")
    d_embeds = [5_000, 2_000, 1000, 500, 250]
    bias = False
    loss_type = "readoff"
    embed = "random"
    title_str = f"W_E={embed}_{bias=}_{n_features=}_{d_mlp=}_{p=}_{loss_type=}"
    ax.set_title(title_str, fontsize=8)
    naive_losses = naive_loss(n_features, d_mlp, p, bias, embed)
    ax.axhline(naive_losses, color="k", linestyle="--", label=f"Naive loss {naive_losses:.2e}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("d_embed")
    ax.set_ylabel("Loss L")
    print(f"Quadrant {bias=} and {embed=}")
    losses = {}
    fixed_random_embedding = embed == "random"  # type: ignore
    fixed_identity_embedding = embed == "identity"  # type: ignore
    for n_steps in [20000]:
        losses[n_steps] = {}
        for d_embed in d_embeds:
            print(f"Run {n_steps} steps, {d_embed} d_embed")
            losses[n_steps][d_embed] = train_on_test_data(
                n_instances=n_instances,
                n_steps=n_steps,
                d_embed=d_embed,
                p=p,
                bias=bias,
                n_features=n_features,
                d_mlp=d_mlp,
                fixed_random_embedding=fixed_random_embedding,
                fixed_identity_embedding=fixed_identity_embedding,
                loss_type=loss_type,  # type: ignore
            )
        plot_loss_curve(ax, losses[n_steps], label=f"{n_steps} steps")
        fig.savefig(out_dir / "loss_scaling_resid_mlp_training_d_embed.png")
    with open(out_dir / f"losses_scale_embed_{title_str}.json", "w") as f:
        json.dump(losses, f)
    # Make plot
    ax.legend(loc="upper center")
    fig.savefig(out_dir / "loss_scaling_resid_mlp_training_d_embed.png")
    print("Saved plot to", out_dir / "loss_scaling_resid_mlp_training_d_embed.png")
    plt.show()

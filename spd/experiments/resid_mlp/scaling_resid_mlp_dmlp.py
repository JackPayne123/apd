import json
import os

import matplotlib.pyplot as plt
import numpy as np

from spd.experiments.resid_mlp.scaling_resid_mlp_training import (
    naive_loss,
    plot_loss_curve,
    train_on_test_data,
)
from spd.settings import REPO_ROOT


def tms_loss(d_embed: int) -> float:
    """Fitting formula, found in scaling_resid_mlp_embed.py"""
    return 0.0486 / d_embed


compute = True

if __name__ == "__main__":
    out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out"
    os.makedirs(out_dir, exist_ok=True)
    n_instances = 20
    n_features = 100
    d_embed = None
    n_steps = 10000
    # Scale d_embed
    p = 0.01
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    fig.suptitle(f"Loss scaling with d_mlp. Using {n_instances} instances")
    bias = False
    loss_type = "readoff"
    embed = "random"
    # d_mlps = [100, 90, 80, 75, 60, 50, 40, 25, 10]
    # d_embeds = [1000, 500]
    # yscale = "linear"
    d_mlps = [100, 90, 75, 50, 25, 10]
    d_embeds = [1000, 500, 250, 125, 100, "100_id", 90, 75, 50]
    yscale = "linear"
    naive_losses = {
        int(d_mlp): naive_loss(n_features, int(d_mlp), p, bias, embed)
        for d_mlp in np.arange(1, 100, 1)
    }
    ax.plot(
        list(naive_losses.keys()),
        list(naive_losses.values()),
        label="naive (monosemantic + d_embed -> inf)",
        color="k",
    )
    # Label 50
    ax.text(
        x=50,
        y=naive_losses[50] * 1.2,
        s=f"d_mlp=50: {naive_losses[50]:.2e}",
        fontsize=6,
        color="k",
    )
    ax.plot([], [], linestyle="--", color="k", label="L_TMS (d_mlp=100)")
    for i in range(len(d_embeds)):
        if d_embeds[i] == "100_id":
            d_embed = 100
            embed = "identity"
        else:
            embed = "random"
            d_embed = int(d_embeds[i])
        fixed_random_embedding = embed == "random"
        fixed_identity_embedding = embed == "identity"
        title_str = f"W_E={embed}_{bias=}_{n_features=}_{p=}_{loss_type}={loss_type}_{n_steps=}"
        losses = {}
        if compute:
            for d_mlp in d_mlps:
                print(f"Run {n_steps} steps, {d_embed} d_embed, {d_mlp} d_mlp")
                losses[d_mlp] = train_on_test_data(
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
            with open(out_dir / f"losses_scale_mlp_{title_str}_d_embed={d_embed}.json", "w") as f:
                json.dump(losses, f)
        else:
            with open(out_dir / f"losses_scale_mlp_{title_str}_d_embed={d_embed}.json") as f:
                losses = json.load(f)
                losses = {
                    int(k): {int(k2): float(v2) for k2, v2 in v.items()} for k, v in losses.items()
                }
        plot_loss_curve(
            ax,
            losses,
            label=f"d_embed = {d_embed} ({embed})",
            fit="linear-parabola",
            fixed_p1=False,
        )
        loss_vals = np.array([v for k, v in losses[50].items()])
        print(loss_vals)
        ax.text(
            55,
            loss_vals.mean(),
            f"d_mlp=50: {loss_vals.mean():.2e} ± {loss_vals.std():.2e}",
            fontsize=6,
            color=f"C{i}",
        )
        ax.axhline(tms_loss(d_embed), linestyle="--", color=f"C{i}")
        ax.set_title(title_str, fontsize=6)
        ax.set_yscale(yscale)
        # ax.set_xscale("log")
        ax.set_xlabel("d_mlp")
        ax.set_ylabel("Loss L")
        fig.savefig(out_dir / "loss_scaling_resid_mlp_training_d_mlp.png")
    ax.legend(loc="upper right")
    fig.savefig(out_dir / "loss_scaling_resid_mlp_training_d_mlp.png")
    print("Saved plot to", out_dir / "loss_scaling_resid_mlp_training_d_mlp.png")
    plt.show()

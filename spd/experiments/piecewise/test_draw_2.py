import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einsum


def plot_single_network(ax, W_in, W_out, max_weight=None):
    """
    Plot a single network on the given axes.

    Args:
    - ax: matplotlib Axes object to plot on
    - W_in: Input weight matrix
    - W_out: Output weight matrix
    - max_weight: Maximum weight for normalization (optional)
    """
    n_features, n_hidden = W_in.shape

    # Normalize weights
    if max_weight is None:
        max_weight = max(W_in.abs().max().item(), W_out.abs().max().item())
    W_in_norm = W_in / max_weight
    W_out_norm = W_out / max_weight

    # Define node positions
    y_input, y_hidden, y_output = 0, -1, -2
    x_input = np.linspace(0.05, 0.95, n_features)
    x_hidden = np.linspace(0.25, 0.75, n_hidden)
    x_output = np.linspace(0.05, 0.95, n_features)

    # Add transparent grey box around hidden layer
    box_width, box_height = 0.8, 0.4
    box = plt.Rectangle(
        (0.5 - box_width / 2, y_hidden - box_height / 2),
        box_width,
        box_height,
        fill=True,
        facecolor="#e4e4e4",
        edgecolor="none",
        alpha=0.33,
    )
    ax.add_patch(box)

    # Plot nodes
    ax.scatter(x_input, [y_input] * n_features, s=100, color="grey", edgecolors="k", zorder=3)
    ax.scatter(x_hidden, [y_hidden] * n_hidden, s=100, color="grey", edgecolors="k", zorder=3)
    ax.scatter(x_output, [y_output] * n_features, s=100, color="grey", edgecolors="k", zorder=3)

    # Plot edges
    cmap = plt.get_cmap("RdBu_r")
    for i in range(n_features):
        for j in range(n_hidden):
            # Input to hidden
            weight = W_in_norm[i, j].item()
            color = cmap(0.5 * (weight + 1))  # Map [-1, 1] to [0, 1]
            ax.plot(
                [x_input[i], x_hidden[j]], [y_input, y_hidden], color=color, linewidth=abs(weight)
            )

            # Hidden to output
            weight = W_out_norm[j, i].item()
            color = cmap(0.5 * (weight + 1))
            ax.plot(
                [x_hidden[j], x_output[i]], [y_hidden, y_output], color=color, linewidth=abs(weight)
            )

    ax.axis("off")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(y_output - 0.5, y_input + 0.5)


def plot_resnet(mlp_components, mlps):
    """
    Plot the entire ResNet visualization.

    Args:
    - mlp_components: List of (W_in, W_out) tuples for each layer and component
    - mlps: List of (W_in_sum, W_out_sum) tuples for each layer
    """
    n_layers = len(mlp_components)
    n_components = mlp_components[0][0].shape[0]  # number of components in each layer

    fig, axs = plt.subplots(
        n_components + 1,  # +1 for the sum of components
        n_layers,
        figsize=(4 * n_layers, 5 * (n_components + 1)),
        constrained_layout=True,
    )
    axs = np.atleast_2d(axs)

    # Find global max weight for consistent normalization
    max_weight = max(
        max(W.abs().max().item() for layer in mlp_components for W in layer),
        max(W.abs().max().item() for layer in mlps for W in layer),
    )

    for layer_idx in range(n_layers):
        for component_idx in range(n_components + 1):
            ax = axs[component_idx, layer_idx]

            if component_idx < n_components:
                W_in, W_out = mlp_components[layer_idx]
                W_in = W_in[component_idx]
                W_out = W_out[component_idx]
            else:
                # Sum of components
                W_in, W_out = mlps[layer_idx]

            plot_single_network(ax, W_in, W_out, max_weight)

            if component_idx == n_components:
                ax.set_title(f"Layer {layer_idx}\n(Sum of components)", fontsize=12)
            elif layer_idx == 0:
                ax.text(
                    -0.1,
                    0.5,
                    f"Component {component_idx}",
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                )

    fig.suptitle("ResNet Visualization", fontsize=16)
    return fig


# Load and process the model
m = torch.load(
    "/data/stefan_heimersheim/projects/SPD/spd/spd/experiments/piecewise/out/plot4sn2l_seed0_topk2.49e-01_topkrecon1.00e+00_topkl2_1.00e+00_lr1.00e-02_bs2000lay2/model_30000.pth"
)

mlp_components = []
mlps = []
for i in range(2):  # Assuming 2 layers
    W_in = einsum(
        m[f"mlps.{i}.linear1.A"], m[f"mlps.{i}.linear1.B"], "embed k, k mlp -> k embed mlp"
    )
    W_out = einsum(
        m[f"mlps.{i}.linear2.A"], m[f"mlps.{i}.linear2.B"], "mlp k, k embed -> k mlp embed"
    )
    mlp_components.append((W_in, W_out))
    mlps.append((W_in.sum(dim=0), W_out.sum(dim=0)))

# Create and display the plot
fig = plot_resnet(mlp_components, mlps)
plt.show()

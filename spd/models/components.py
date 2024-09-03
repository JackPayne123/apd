import torch
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.utils import init_param_


class ParamComponents(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
    ):
        """
        Args:
            in_dim: Input dimension of the parameter to be replaced with subnetworks
            out_dim: Output dimension of the parameter to be replaced with subnetworks
            k: Number of subnetworks.
            DAN: resid_component and resid_dim deleted here. I am not sure what the point of them was in
            the first place, so you may want to add them back in.
        """
        super().__init__()

        self.subnetworks = torch.empty(k, in_dim, out_dim)
        init_param_(self.subnetworks)

    def forward(
        self,
        x: Float[Tensor, "... dim1"],
    ) -> tuple[Float[Tensor, "... dim2"], Float[Tensor, "... k"]]:
        inner_acts = einsum(x, self.subnetworks, "batch dim1, k dim1 dim2 ->batch k dim2")
        out = einsum(inner_acts, "batch k dim2 -> batch dim2")
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... dim1"],
        topk_mask: Bool[Tensor, "... k"],
    ) -> tuple[Float[Tensor, "... dim2"], Float[Tensor, "... k"]]:
        """
        Performs a forward pass using only the top-k subnetwork activations.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetwork activations to keep.

        Returns:
            out: Output tensor
            inner_acts: Subnetwork activations
        """

        inner_acts = einsum(x, self.subnetworks, "batch dim1, k dim1 dim2 ->batch k dim2")
        # Dan: I am assuming no n_instances here. I think that's correct but lmk if not
        inner_acts_topk = einsum(inner_acts, topk_mask, "batch k dim2, batch k -> batch k dim2")
        out = einsum(inner_acts_topk, "batch k dim2 -> batch dim2")
        return out, inner_acts_topk


class MLPComponents(nn.Module):
    """
    A module that contains two linear layers with a ReLU activation in between.

    Note that the first linear layer has a bias that is not decomposed, and the second linear layer
    has no bias.
    """

    def __init__(
        self,
        d_embed: int,
        d_mlp: int,
        k: int,
        input_bias: Float[Tensor, " d_mlp"] | None = None,
    ):
        super().__init__()
        self.linear1 = ParamComponents(d_embed, d_mlp, k)
        self.bias1 = nn.Parameter(torch.zeros(d_mlp))
        if input_bias is not None:
            self.bias1.data = input_bias.detach().clone()
        self.linear2 = ParamComponents(d_mlp, d_embed, k)

    def forward(
        self, x: Float[Tensor, "... d_embed"]
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []
        x, inner_acts_linear1 = self.linear1(x)
        x += self.bias1
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x, inner_acts_linear2 = self.linear2(
            F.relu(x)
        )  # is there a reason we aren't using self.relu = nn.ReLU() here?
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)
        return x, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... d_embed"],
        topk_mask: Bool[Tensor, "... k"],
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each linear layer.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which components to keep.
        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []

        # First linear layer
        x, inner_acts_linear1 = self.linear1.forward_topk(x, topk_mask)
        x += self.bias1
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = F.relu(x)

        # Second linear layer
        x, inner_acts_linear2 = self.linear2.forward_topk(x, topk_mask)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)

        return x, layer_acts, inner_acts

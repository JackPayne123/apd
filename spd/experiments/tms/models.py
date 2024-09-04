import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.models.base import Model, SPDModel
from spd.types import RootPath


class TMSModel(Model):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.empty((n_instances, n_features, n_hidden), device=device))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(torch.zeros((n_instances, n_features), device=device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = torch.einsum("...if,ifh->...ih", features, self.W)
        out = torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out

    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        """List of all parameters which will be decomposed with SPD."""
        return [self.W, rearrange(self.W, "i f h -> i h f")]


class TMSSPDModel(SPDModel):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        k: int | None,
        bias_val: float,
        train_bias: bool,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.k = k if k is not None else n_features
        self.bias_val = bias_val
        self.train_bias = train_bias

        self.subnetworks = nn.Parameter(
            torch.empty((n_instances, self.k, n_features, n_hidden), device=device)
        )

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data) if train_bias else bias_data

        nn.init.xavier_normal_(self.subnetworks)

        self.n_param_matrices = 2  # Two W matrices (even though they're tied)

    def all_subnetworks(self) -> list[Float[Tensor, "n_instances k n_features n_hidden"]]:
        return [self.subnetworks, rearrange(self.subnetworks, "i k f h -> i k h f")]

    def forward(
        self, x: Float[Tensor, "... n_inst n_feat"]
    ) -> tuple[
        Float[Tensor, "... n_inst n_feat"],
        list[Float[Tensor, "... n_inst n_feat"]],
        list[Float[Tensor, "... n_inst k"]],
    ]:
        inner_act_0 = einsum(
            x,
            self.subnetworks,
            "... n_inst n_feat, n_inst k n_feat n_hidden -> ... n_inst k n_hidden",
        )
        layer_act_0 = einsum(inner_act_0, "... n_inst k n_hidden -> ... n_inst n_hidden")

        inner_act_1 = einsum(
            layer_act_0,
            self.subnetworks,
            "... n_inst n_hidden, n_inst k n_feat n_hidden -> ... n_inst k n_feat",
        )
        layer_act_1 = einsum(inner_act_1, "... n_inst k n_feat -> ... n_inst n_feat")

        pre_relu = layer_act_1 + self.b_final

        out = F.relu(pre_relu)
        # Can pass layer_act_1 or pre_relu to layer_acts[1] as they're the same for the gradient
        # operations we care about (dout/d(inner_act_1)).
        return out, [layer_act_0, layer_act_1], [inner_act_0, inner_act_1]

    def forward_topk(
        self,
        x: Float[Tensor, "... i f"],
        topk_mask: Bool[Tensor, "... n_inst k"],
    ) -> tuple[
        Float[Tensor, "... i f"],
        list[Float[Tensor, "... i f"]],
        list[Float[Tensor, "... i k"]],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        inner_act_0 = einsum(
            x,
            self.subnetworks,
            "... n_inst n_feat, n_inst k n_feat n_hidden -> ... n_inst k n_hidden",
        )
        assert topk_mask.shape == inner_act_0.shape[:-1]
        inner_act_0_topk = einsum(
            inner_act_0, topk_mask, "... n_inst k n_hidden, ... n_inst k -> ... n_inst k n_hidden"
        )
        layer_act_0 = einsum(inner_act_0_topk, "... n_inst k n_hidden -> ... n_inst n_hidden")

        inner_act_1 = einsum(
            layer_act_0,
            self.subnetworks,
            "... n_inst n_hidden, n_inst k n_feat n_hidden -> ... n_inst k n_feat",
        )

        assert topk_mask.shape == inner_act_1.shape[:-1]
        inner_act_1_topk = einsum(
            inner_act_1, topk_mask, "... n_inst k n_feat, ... n_inst k -> ... n_inst k n_feat"
        )
        layer_act_1 = einsum(inner_act_1_topk, "... n_inst k n_feat -> ... n_inst n_feat")

        pre_relu = layer_act_1 + self.b_final
        out = F.relu(pre_relu)
        return out, [layer_act_0, layer_act_1], [inner_act_0_topk, inner_act_1_topk]

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDModel":  # type: ignore
        pass

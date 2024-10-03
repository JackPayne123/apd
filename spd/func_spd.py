from functools import partial

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# torch.set_float32_matmul_precision("high")


# Define a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 16),  # Input layer
            nn.ReLU(),
            nn.Linear(16, 2),  # Output layer
        )

    def forward(self, x: Float[Tensor, " ... input_dim"]):
        return self.layers(x)


def calc_topk_mask(scores: Float[Tensor, "batch k"], topk: int) -> Float[Tensor, "batch k"]:
    topk_indices = scores.topk(topk, dim=-1).indices
    topk_mask = torch.zeros_like(scores, dtype=torch.bool)
    topk_mask.scatter_(dim=-1, index=topk_indices, value=True)
    return topk_mask


model = SimpleMLP()

# Generate random data without labels
X = torch.randn(1000, 3)  # 100 samples, 10 features

# Create a DataLoader without labels
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


def optimize(
    pretrained_model: nn.Module,
    k: int,
    topk: int,
    dataloader: DataLoader[Float[Tensor, " n_inputs"]],
    lr: float = 1e-3,
    steps: int = 1000,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device=device)

    pretrained_params = pretrained_model.state_dict()

    decomposable_params = pretrained_params.keys()

    k_params = {}
    for key, value in pretrained_params.items():
        if key in decomposable_params:
            shape = list(value.shape)
            k_params[key] = torch.empty([k, *shape], device=device, requires_grad=True)
            torch.nn.init.kaiming_uniform_(k_params[key], a=3)

    opt = torch.optim.AdamW(k_params.values(), lr=lr, weight_decay=0.0)

    alpha = torch.ones(k, device=device)

    for batch in tqdm(dataloader):
        batch = batch[0].to(device=device)
        print("Running pretrained model")
        pretrained_out = pretrained_model(batch)

        print("Compiling test function")

        def calc_jacobian(alpha: Float[Tensor, "k"]) -> Float[Tensor, "batch n_outputs k"]:
            return torch.autograd.functional.jacobian(
                lambda alpha: functional_call(
                    pretrained_model,
                    {k: einops.einsum(v, alpha, "k ..., k -> ...") for k, v in k_params.items()},
                    batch,
                    # is it bad that function doesn't bind loop variable batch? I don't think so
                ),
                alpha,
            ).squeeze(dim=-2)

        # calc_jacobian_compiled = torch.compile(calc_jacobian)

        jacobian = calc_jacobian(alpha)
        # print(f"Jacobian: {(jacobian**2).sum()}")

        # def test_func(alpha):
        #     return functional_call(
        #         pretrained_model,
        #         {k: einops.einsum(v, alpha, "k ..., k -> ...") for k, v in k_params.items()},
        #         batch,
        #     )

        # opt_test_func = torch.compile(test_func)
        # print("Calculating jacobian")
        # jacobian = torch.autograd.functional.jacobian(
        #     opt_test_func,
        #     alpha,
        # ).squeeze(dim=-2)

        # jacobian = torch.autograd.functional.jacobian(
        #     lambda alpha: functional_call(
        #         pretrained_model,
        #         {k: einops.einsum(v, alpha, "k ..., k -> ...") for k, v in k_params.items()},
        #         batch,
        #     ),
        #     alpha,
        # ).squeeze(dim=-2)
        # print("Calculating topk mask")
        attribs: Float[Tensor, "batch k"] = einops.reduce(
            jacobian**2, "batch n_outputs k -> batch k", "sum"
        )
        topk_mask = calc_topk_mask(attribs, topk).float()
        print(f"Attribs: {attribs.sum()}")

        # print("Calculating per sample topk forward")

        def per_sample_topk_forward(
            batch_i: Float[Tensor, " n_inputs"],
            topk_mask_i: Float[Tensor, " key"],
            k_params: dict[str, Float[Tensor, " ... k"]],
        ):
            masked_params = {
                key: einops.einsum(value, topk_mask_i, "k ..., k -> ...")
                for key, value in k_params.items()
            }
            return functional_call(pretrained_model, masked_params, batch_i)

        per_sample_topk_forward_p = partial(per_sample_topk_forward, k_params=k_params)
        out_recon = torch.vmap(per_sample_topk_forward_p)(batch, topk_mask)
        recon_loss = (out_recon - pretrained_out).pow(2).mean()

        param_match_loss = 0.0
        for key, value in k_params.items():
            param_match_loss += (value - pretrained_params[key]).pow(2).mean()

        l2_loss = 0.0
        for _, value in k_params.items():
            l2_loss += (
                einops.einsum(
                    topk_mask,
                    value,
                    "b k, k ... -> ...",
                )
                .pow(2)
                .mean()
            )

        loss = recon_loss + param_match_loss + l2_loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"Loss: {loss.item()} Attribs: {attribs.sum()}")


optimize(pretrained_model=model, k=10, topk=3, dataloader=dataloader)

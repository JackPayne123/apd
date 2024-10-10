import json
from functools import partial
from pathlib import Path

import einops
import fire
import matplotlib.pyplot as plt
import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.bigrams.model import BigramDataset, BigramModel

# %%


A_vocab_size = 100  # A ranges from 0 to 99
B_vocab_size = 5  # B ranges from 0 to 4
embedding_dim = 200
hidden_dim = 200
batch_size = 500

dataset = BigramDataset(A_vocab_size, B_vocab_size)

model = BigramModel(dataset.n_A, dataset.n_B, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("bigram_model.pt", weights_only=True))
model.to("cpu")

k = 5

# L2 norm of all weights
l2_norm = sum(p.norm().pow(2) for p in model.parameters())
print("L2 norm of all weights:", l2_norm.item())

# L2 norm of all weights
l2_norm = sum(p.pow(2).mean() for p in model.parameters())
print("L2 norm of all weights:", l2_norm.item())

l2_norm = 0
n_params = 0
for p in model.parameters():
    l2_norm += p.norm().pow(2)
    n_params += p.numel()
print("L2 norm of all weights:", l2_norm.item() / n_params)

k_params = torch.load("out/k_params_step_10000.pt", weights_only=True)
for key, param in k_params.items():
    print(key, param.shape)

l2_norm_by_k = []
for ki in range(k):
    l2_norm = sum(p[ki].norm().pow(2) for p in k_params.values())
    l2_norm_by_k.append(l2_norm.item())
print("L2 norm by k:", l2_norm_by_k)


l2_norm_by_k = []
for ki in range(k):
    l2_norm = sum(p[ki].pow(2).mean() for p in k_params.values())
    l2_norm_by_k.append(l2_norm.item())
print("L2 norm by k:", l2_norm_by_k)

l2_norm_by_k = []
for ki in range(k):
    l2_norm = 0
    n_params = 0
    for p in k_params.values():
        l2_norm += p[ki].norm().pow(2)
        n_params += p[ki].numel()
    l2_norm_by_k.append(l2_norm.item() / n_params)
print("L2 norm by k:", l2_norm_by_k)


def functional_model(x, k: int, k_params=k_params):
    params = {key: v[k].to("cpu") for key, v in k_params.items()}
    return torch.func.functional_call(model, params, (x,))


def full_model(x, k_params=k_params):
    params = {key: v.sum(dim=0).to("cpu") for key, v in k_params.items()}
    return torch.func.functional_call(model, params, (x,))


# %%
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for i, (batch, _) in enumerate(dataloader):
    fig, axes = plt.subplots(nrows=k + 1, constrained_layout=True, figsize=(6, 6))
    out_full = full_model(batch)[0]
    ylim = out_full.detach().cpu().numpy().min(), out_full.detach().cpu().numpy().max()
    axes[0].plot(out_full.detach().cpu().numpy(), label="full model", alpha=1, color="k", ls=":")
    for ki in range(k):
        out = functional_model(batch, ki)[0]
        loss = mse_loss(out, out_full)
        fig.suptitle(f"Input B = {batch[0, -5:]}")
        # print("k =", ki, "functional_model(batch, k) =", out)
        axes[ki + 1].plot(
            out.detach().cpu().numpy(),
            # label=f"k = {ki}, mse = {loss.item():.3e}",
            alpha=0.3,
        )
        axes[ki + 1].set_ylim(ylim)
        # axes[ki + 1].set_xlim(1, 60)
        axes[ki + 1].set_title(f"k = {ki}, mse = {loss.item():.3e}")

    fig.legend(loc="lower right")
    if i > 10:
        break
# %%
batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
batch, label = next(iter(dataloader))
out_full = full_model(batch)
out_ki = []
for ki in range(k):
    out_ki.append(functional_model(batch, ki))

image = torch.zeros((5, 100))
for i in range(500):
    a = torch.argmax(batch[i][:100])
    b = torch.argmax(batch[i][100:])
    for ki in range(k):
        loss = mse_loss(out_ki[ki][i], out_full[i])
        if loss.item() < 1e-5:
            print(f"i = {i} uses k = {ki} (a={a}, b={b})")
            image[b, a] = ki

plt.matshow(image)
plt.show()
# %%

batch_size = 1
for a in range(A_vocab_size):
    for b in range(B_vocab_size):
        batch = torch.zeros(105)
        batch[a] = 1
        batch[100 + b] = 1
        out_full = full_model(batch)[0]
        for ki in range(k):
            out = functional_model(batch, ki)[0]
            loss = mse_loss(out, out_full)
            print(f"k = {ki}, mse = {loss.item():.3e}")

# %%
# W_E = k_params["W_E"].detach().cpu().numpy()
# fig, ax = plt.subplots()
# for k in range(5):
#     for j in range(5):
#         color = f"C{k}"
#         alpha = 1 if k == j else 0.3
#         ax.scatter(W_E[k, 100 + j, 0], W_E[k, 100 + j, 1], color=color, alpha=alpha)
#     # for i in range(100):
#     #     ax.scatter(W_E[k, i, 0], W_E[k, i, 1], color="black", marker="x")
# ax.legend()
# plt.show()

# %%

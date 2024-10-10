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

# L2 norm of all weights
l2_norm = sum(p.norm().pow(2) for p in model.parameters())
print("L2 norm of all weights:", l2_norm.item())

# L2 norm of all weights
l2_norm = sum(p.pow(2).mean() for p in model.parameters())
print("L2 norm of all weights:", l2_norm.item())

k_params = torch.load("out/k_params_step_10000.pt", weights_only=True)
for key, param in k_params.items():
    print(key, param.shape)

l2_norm_by_k = []
for k in range(5):
    l2_norm = sum(p[k].norm().pow(2) for p in k_params.values())
    l2_norm_by_k.append(l2_norm.item())
print("L2 norm by k:", l2_norm_by_k)


l2_norm_by_k = []
for k in range(5):
    l2_norm = sum(p[k].pow(2).mean() for p in k_params.values())
    l2_norm_by_k.append(l2_norm.item())
print("L2 norm by k:", l2_norm_by_k)


def functional_model(x, k: int, k_params=k_params):
    params = {key: v[k].to("cpu") for key, v in k_params.items()}
    return torch.func.functional_call(model, params, (x,))


batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for i, (batch, label) in enumerate(dataloader):
    fig, ax = plt.subplots(constrained_layout=True)
    for k in range(5):
        fig.suptitle(f"Input B = {batch[0, -5:]}")
        out = functional_model(batch, k)[0]
        print("k =", k, "functional_model(batch, k) =", out)
        ax.plot(out.detach().cpu().numpy(), label=f"Subnet k = {k}", alpha=0.3)
    fig.legend()
    if i > 5:
        break
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

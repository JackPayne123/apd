import json
from functools import partial
from pathlib import Path

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
batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = BigramModel(dataset.n_A, dataset.n_B, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("bigram_model.pt", weights_only=True))

k_params = torch.load("out/k_params_step_6000.pt", weights_only=True)
for key, param in k_params.items():
    print(key, param.shape)
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

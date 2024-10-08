import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from einops import einsum
from jaxtyping import Float, Int
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Create PyTorch Dataset class
class BigramDataset(Dataset[tuple[Int[Tensor, "2"], Int[Tensor, ""]]]):
    def __init__(self, n_A: int, n_B: int):
        self.n_A = n_A
        self.n_B = n_B
        self.n_C = n_A * n_B

        # Generate rules
        lookup_table = torch.zeros(n_A, n_B, dtype=torch.long)
        for a in range(n_A):
            for b in range(n_B):
                lookup_table[a, b] = a * n_B + b

        data, labels = [], []
        for a in range(n_A):
            for b in range(n_B):
                a_vec = torch.nn.functional.one_hot(torch.tensor(a), num_classes=n_A)
                b_vec = torch.nn.functional.one_hot(torch.tensor(b), num_classes=n_B)
                data.append(torch.cat([a_vec, b_vec], dim=0))
                labels.append(torch.nn.functional.one_hot(lookup_table[a, b], num_classes=self.n_C))

        self.data = torch.stack(data)
        self.labels = torch.stack(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

    def get_vocab_sizes(self) -> tuple[int, int, int]:
        return self.n_A, self.n_B, self.n_C


# Define the neural network
class BigramModel(nn.Module):
    def __init__(self, n_A: int, n_B: int, d_embed: int, d_hidden: int):
        super().__init__()
        self.n_A = n_A
        self.n_B = n_B
        self.n_C = n_A * n_B
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.W_E_A = nn.Parameter(torch.randn(self.n_A, self.d_embed))
        self.W_E_B = nn.Parameter(torch.randn(self.n_B, self.d_embed))
        self.fc1 = nn.Linear(self.d_embed, self.d_hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.d_hidden, self.d_embed)
        self.unembed = nn.Linear(self.d_embed, self.n_C)

    def forward(
        self, inputs: Int[Tensor, "batch_size n_A + n_B"]
    ) -> Float[Tensor, "batch_size d_embed"]:
        xA = einsum(
            self.W_E_A,
            inputs[:, 0:100].float(),
            "n_A d_embed, batch_size n_A -> batch_size d_embed",
        )
        xB = einsum(
            self.W_E_B,
            inputs[:, 100:105].float(),
            "n_B d_embed, batch_size n_B -> batch_size d_embed",
        )
        x = xA + xB
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.unembed(x)
        return x

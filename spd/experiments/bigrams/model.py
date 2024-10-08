import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset


# Create PyTorch Dataset class
class BigramDataset(Dataset[tuple[Int[Tensor, "2"], Int[Tensor, ""]]]):
    def __init__(self, n_A: int, n_B: int):
        self.n_A = n_A
        self.n_B = n_B
        self.output_vocab_size = n_A * n_B

        # Generate rules
        lookup_table = torch.zeros(n_A, n_B, dtype=torch.long)
        for a in range(n_A):
            for b in range(n_B):
                lookup_table[a, b] = a * n_B + b

        data, labels = [], []
        for a in range(n_A):
            for b in range(n_B):
                data.append([a, b])
                labels.append(lookup_table[a, b])

        self.data = torch.tensor(data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

    def get_vocab_sizes(self) -> tuple[int, int, int]:
        return self.n_A, self.n_B, self.output_vocab_size


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
        self.embed = lambda x: self.W_E_A[x[:, 0]] + self.W_E_B[x[:, 1]]
        self.fc1 = nn.Linear(self.d_embed, self.d_hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.d_hidden, self.d_embed)
        self.unembed = nn.Linear(self.d_embed, self.n_C)

    def forward(self, inputs: Int[Tensor, "batch_size 2"]) -> Float[Tensor, "batch_size d_embed"]:
        x = self.embed(inputs)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.unembed(x)
        return x

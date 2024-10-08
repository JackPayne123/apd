import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset


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
        self.W_E = nn.Parameter(torch.empty(self.n_A + self.n_B, self.d_embed))
        torch.nn.init.xavier_uniform_(self.W_E)
        self.fc1 = nn.Linear(self.d_embed, self.d_hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.d_hidden, self.d_embed)
        self.unembed = nn.Linear(self.d_embed, self.n_C)

    def forward(self, inputs: Int[Tensor, "batch_size n_C"]) -> Float[Tensor, "batch_size d_embed"]:
        # print(f"{inputs.shape=}")
        # print(f"{self.W_E.shape=}")
        x = einsum(
            self.W_E,
            inputs.float(),
            "n_C d_embed, batch_size n_C -> batch_size d_embed",
        )
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.unembed(x)
        return x

from collections.abc import Callable
from typing import Any

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.bigrams.model import BigramDataset, BigramModel


def train(
    A_vocab_size: int = 100,
    B_vocab_size: int = 5,
    embedding_dim: int = 200,
    hidden_dim: int = 200,
    epochs: int = 2000,
    learning_rate: float = 0.001,
    seed: int = 0,
    activation: bool = True,
    batch_size: int | None = None,
    log: Callable[[dict[str, Any]], None] = lambda d: tqdm.write(str(d)),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = BigramDataset(A_vocab_size, B_vocab_size)
    batch_size = batch_size or len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BigramModel(dataset.n_A, dataset.n_B, embedding_dim, hidden_dim, activation=activation)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs: Float[torch.Tensor, "batch B_vocab_size"] = model(batch_inputs)
            loss: Float[torch.Tensor, ""] = criterion(outputs, batch_targets.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if epoch % 100 == 0:
            log({"epoch": epoch, "loss": avg_loss})

    torch.save(model.state_dict(), "bigram_model.pt")


if __name__ == "__main__":
    fire.Fire(train)

# %%

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.bigrams.model import BigramDataset, BigramModel

# Set random seed for reproducibility
# torch.manual_seed(0)
# np.random.seed(0)

# Parameters
A_vocab_size = 100  # A ranges from 0 to 99
B_vocab_size = 5  # B ranges from 0 to 4
embedding_dim = 20
hidden_dim = 50
epochs = 2000
learning_rate = 0.01


dataset = BigramDataset(A_vocab_size, B_vocab_size)
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = BigramModel(dataset.n_A, dataset.n_B, embedding_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in tqdm(range(1, epochs + 1)):
    total_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs: Float[torch.Tensor, "batch B_vocab_size"] = model(batch_inputs)
        loss: Float[torch.Tensor, ""] = criterion(outputs, batch_targets.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    if epoch % 100 == 0:
        tqdm.write(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.3e}")

# Save model
torch.save(model.state_dict(), "bigram_model.pt")

# %%

# Load model
new_model = BigramModel(dataset.n_A, dataset.n_B, embedding_dim, hidden_dim)
new_model.load_state_dict(torch.load("bigram_model.pt", weights_only=True))
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for batch_inputs, batch_targets in dataloader:
    outputs: Float[torch.Tensor, "batch B_vocab_size"] = new_model(batch_inputs)
    loss: Float[torch.Tensor, ""] = criterion(outputs, batch_targets.float())
    print(loss.item())

# %%

# Visualize embeddings coloured by B
# Get embeddings + colors for all 500 inputs
embeddings = []
As = []
Bs = []
for a in range(A_vocab_size):
    for b in range(B_vocab_size):
        x: Float[torch.Tensor, " embedding_dim"] = model.W_E[a] + model.W_E[b + A_vocab_size]
        embeddings.append(x.detach().cpu())
        As.append(a)
        Bs.append(b)
embeddings = np.array(embeddings)
As = np.array(As)
Bs = np.array(Bs)
# PCA
pca = PCA(n_components=10)
embeddings_3d = pca.fit_transform(embeddings)

# Figure for Bs coloring
fig_b = go.Figure(
    data=[
        go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode="markers",
            marker=dict(size=5, color=Bs, colorscale="Viridis", opacity=0.8),
        )
    ]
)

fig_b.update_layout(
    title="3D Embedding Visualization (Colored by B)",
    scene=dict(
        xaxis_title="Embedding Dimension 1",
        yaxis_title="Embedding Dimension 2",
        zaxis_title="Embedding Dimension 3",
    ),
    width=900,
    height=700,
)

fig_b.show()

# Figure for As coloring
fig_a = go.Figure(
    data=[
        go.Scatter3d(
            x=embeddings_3d[:, 0 + 5],
            y=embeddings_3d[:, 1 + 5],
            z=embeddings_3d[:, 2 + 5],
            mode="markers",
            marker=dict(size=5, color=As, colorscale="Plasma", opacity=0.8),
        )
    ]
)

fig_a.update_layout(
    title="3D Embedding Visualization (Colored by A)",
    scene=dict(
        xaxis_title="Embedding Dimension 1",
        yaxis_title="Embedding Dimension 2",
        zaxis_title="Embedding Dimension 3",
    ),
    width=900,
    height=700,
)

fig_a.show()

# %%

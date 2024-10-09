import os

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# Custom Dataset class for Optical Recognition of Handwritten Letters
class OpticalLettersDataset(Dataset):
    def __init__(self, file_path: str):
        # Load the dataset
        data = pd.read_csv(file_path, header=None)

        # Extract features and labels
        self.features = data.iloc[:, 1:].values.astype(np.float32)
        self.labels = data.iloc[:, 0].values

        # Encode the labels (A-Z) as integers (0-25)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


# MLP Model Definition
class OpticalLettersModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(OpticalLettersModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out


def train(
    data_file: str = "./data/letter-recognition.data",
    input_size: int = 16,
    hidden_size: int = 512,
    num_classes: int = 26,  # 26 uppercase letters (A-Z)
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    seed: int = 0,
    split_ratio: float = 0.8,
    log_interval: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load the dataset
    full_dataset = OpticalLettersDataset(file_path=data_file)

    # Split dataset into training and validation sets
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model, loss function, optimizer
    model = OpticalLettersModel(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]")
        for features, labels in loop:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total

        if epoch % log_interval == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], "
                f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

    # Save the model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/optical_letters_model.pth")


if __name__ == "__main__":
    fire.Fire(train)

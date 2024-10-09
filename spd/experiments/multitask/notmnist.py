import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from typing import Callable, Any
import fire

# Custom Dataset class for NotMNIST
class NotMNISTDataset(Dataset):
    def __init__(self, root: str):
        self.transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
        self.data = []
        self.targets = []
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Load data and labels from class folders
        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.data.append(img_path)
                self.targets.append(self.class_to_idx[cls])
        
        print(f"Loaded {len(self.data)} images from {root}, split into {len(self.classes)} classes")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        img_path = self.data[idx]
        label = self.targets[idx]
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        except OSError:
            # Skip corrupted images
            return self.__getitem__((idx + 1) % len(self.data))
        if self.transform:
            image = self.transform(image)
        return image, label

# MLP Model Definition
class NotMNISTModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.input_size = 28 * 28
        self.num_classes = 10
        self.hidden_size = hidden_size
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out

def train(
    root: str = './data/notMNIST_small',
    hidden_size: int = 512,
    num_epochs: int = 2,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    seed: int = 0,
    split_ratio: float = 0.8,
    log_interval: int = 1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Load entire dataset
    dataset = NotMNISTDataset(root=root)

    # Split dataset into training and validation sets
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, optimizer
    model = NotMNISTModel(hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{num_epochs}]')
        for images, labels in loop:
            images = images.view(-1, model.input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
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
            loop.set_postfix(loss=loss.item(), accuracy=100.*correct/total)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        if epoch % log_interval == 0:
            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Save the model checkpoint
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/notmnist_model.pth')

if __name__ == "__main__":
    fire.Fire(train)

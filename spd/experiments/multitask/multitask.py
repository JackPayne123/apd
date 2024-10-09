import os
import random

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


class MultiTaskDataset(Dataset):
    def __init__(self, datasets, input_sizes, num_classes, p_active=0.5):
        """
        datasets: List of individual PyTorch datasets.
        input_sizes: List of input sizes corresponding to each dataset.
        num_classes: List of number of classes for each task.
        p_active: Probability of each task being active.
        """
        self.datasets = datasets
        self.input_sizes = input_sizes
        self.num_classes = num_classes
        self.p_active = p_active
        self.max_length = max(len(ds) for ds in datasets)
        self.num_tasks = len(datasets)

    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        # For each task, decide whether it's active
        active_tasks = []
        inputs = []
        labels = []
        for i in range(self.num_tasks):
            is_active = random.random() < self.p_active
            if is_active:
                # Sample a data point from the respective dataset
                dataset = self.datasets[i]
                # Handle index overflow
                data_idx = random.randint(0, len(dataset) - 1)
                data, label = dataset[data_idx]
                # Flatten the input if necessary
                if isinstance(data, torch.Tensor) and data.dim() > 1:
                    data = data.view(-1)
                inputs.append(data)
                labels.append(label)
                active_tasks.append(1)
            else:
                # Use zero input for inactive tasks
                input_size = self.input_sizes[i]
                inputs.append(torch.zeros(input_size))
                labels.append(-1)  # Use -1 to indicate inactive task
                active_tasks.append(0)

        # Concatenate inputs
        concatenated_input = torch.cat(inputs)
        # Convert labels to tensor
        labels = torch.tensor(labels)
        active_tasks = torch.tensor(active_tasks, dtype=torch.bool)
        return concatenated_input, labels, active_tasks


class MultiTaskModel(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_classes):
        """
        input_sizes: List of input sizes corresponding to each task.
        hidden_size: Size of hidden layers.
        num_classes: List of number of classes for each task.
        """
        super(MultiTaskModel, self).__init__()
        self.input_sizes = input_sizes
        self.num_tasks = len(input_sizes)
        total_input_size = sum(input_sizes)
        self.shared_layers = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Create separate output layers for each task
        self.output_layers = nn.ModuleList(
            [nn.Linear(hidden_size, num_classes[i]) for i in range(self.num_tasks)]
        )

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(shared_representation))
        return outputs  # List of outputs for each task


def train(
    tasks=["mnist", "fashionmnist", "kmnist", "notmnist"],
    p_active=0.5,
    hidden_size=512,
    num_epochs=10,
    batch_size=128,
    learning_rate=0.001,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Prepare datasets and input sizes
    datasets_list = []
    input_sizes = []
    num_classes = []
    for task in tasks:
        if task == "mnist":
            dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            input_size = 28 * 28
            n_classes = 10
        elif task == "fashionmnist":
            dataset = datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform
            )
            input_size = 28 * 28
            n_classes = 10
        elif task == "kmnist":
            dataset = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
            input_size = 28 * 28
            n_classes = 10
        elif task == "notmnist":
            # Assuming NotMNISTDataset is defined in notmnist.py
            from notmnist import NotMNISTDataset

            dataset = NotMNISTDataset(root="./data/notMNIST_small")
            input_size = 28 * 28
            n_classes = 10
        elif task == "uciletters":
            # Assuming OpticalLettersDataset is defined in uciletters.py
            from uciletters import OpticalLettersDataset

            dataset = OpticalLettersDataset(file_path="./data/letter-recognition.data")
            input_size = 16  # As per dataset
            n_classes = 26
        else:
            raise ValueError(f"Unknown task: {task}")
        datasets_list.append(dataset)
        input_sizes.append(input_size)
        num_classes.append(n_classes)

    # Create the MultiTaskDataset
    multi_task_dataset = MultiTaskDataset(
        datasets=datasets_list, input_sizes=input_sizes, num_classes=num_classes, p_active=p_active
    )

    # DataLoader
    dataloader = DataLoader(multi_task_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = MultiTaskModel(
        input_sizes=input_sizes, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create directory for models
    os.makedirs("models", exist_ok=True)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        correct_counts = [0] * len(tasks)
        total_counts = [0] * len(tasks)
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]")
        for inputs, labels, active_tasks in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            active_tasks = active_tasks.to(device)
            outputs = model(inputs)
            loss = 0
            # Compute loss only for active tasks
            for i in range(len(tasks)):
                task_active = active_tasks[:, i]
                if task_active.any():
                    task_indices = torch.where(task_active)[0]
                    task_outputs = outputs[i][task_indices]
                    task_labels = labels[task_indices, i]
                    task_loss = F.cross_entropy(task_outputs, task_labels)
                    loss += task_loss
                    # Compute accuracy
                    preds = task_outputs.argmax(dim=1)
                    correct = preds.eq(task_labels).sum().item()
                    correct_counts[i] += correct
                    total_counts[i] += task_labels.size(0)
            if loss == 0:
                continue  # No active tasks in this batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Update progress bar
            loop.set_postfix(loss=loss.item())

        # Print epoch results
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
        for i, task in enumerate(tasks):
            if total_counts[i] > 0:
                accuracy = 100.0 * correct_counts[i] / total_counts[i]
                print(f"Task: {task}, Accuracy: {accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "models/multitask_model.pth")


if __name__ == "__main__":
    fire.Fire(train)

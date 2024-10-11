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

from spd.experiments.multitask.single import transform


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
        target_probs = []
        for i in range(self.num_tasks):
            is_active = random.random() < self.p_active
            if is_active:
                # Sample a data point from the respective dataset
                dataset = self.datasets[i]
                data_idx = random.randint(0, len(dataset) - 1)
                data, label = dataset[data_idx]
                # Flatten the input if necessary
                if isinstance(data, torch.Tensor) and data.dim() > 1:
                    data = data.view(-1)
                inputs.append(data)

                # Create one-hot encoded label
                one_hot_label = torch.zeros(self.num_classes[i])
                one_hot_label[label] = 1.0
                target_probs.append(one_hot_label)

                active_tasks.append(1)
            else:
                # Use zero input for inactive tasks
                input_size = self.input_sizes[i]
                inputs.append(torch.zeros(input_size))
                # Return uniform distribution for inactive task
                uniform_prob = torch.ones(self.num_classes[i]) / self.num_classes[i]
                target_probs.append(uniform_prob)
                active_tasks.append(0)

        # Concatenate inputs
        concatenated_input = torch.cat(inputs)
        # Convert labels and active_tasks to tensors
        target_probs = torch.stack(target_probs)
        active_tasks = torch.tensor(active_tasks, dtype=torch.bool)
        return concatenated_input, target_probs, active_tasks


class MultiTaskModel(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_classes):
        """
        input_sizes: List[int], input sizes for each task.
        hidden_size: int, size of hidden layers.
        num_classes: List[int], number of classes for each task.
        """
        super(MultiTaskModel, self).__init__()
        self.input_sizes = input_sizes
        self.num_classes = num_classes
        self.num_tasks = len(input_sizes)
        self.total_input_size = sum(input_sizes)
        self.total_output_size = sum(num_classes)

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.total_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Shared output layer
        self.output_layer = nn.Linear(hidden_size, self.total_output_size)

        # Compute output indices for slicing
        self.output_indices = self.compute_output_indices()

    def compute_output_indices(self):
        """
        Computes start and end indices for each task's outputs in the combined output tensor.
        """
        indices = []
        current_idx = 0
        for n_class in self.num_classes:
            start_idx = current_idx
            end_idx = current_idx + n_class
            indices.append((start_idx, end_idx))
            current_idx = end_idx
        return indices

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, total_input_size]
        Returns:
            outputs: List of softmax probability distributions for each task.
        """
        shared_representation = self.shared_layers(x)
        raw_outputs = self.output_layer(shared_representation)
        outputs = []

        # Apply softmax to each task's output
        for i in range(self.num_tasks):
            output_start, output_end = self.output_indices[i]
            if raw_outputs.ndim == 2:
                task_output = raw_outputs[:, output_start:output_end]
            elif raw_outputs.ndim == 1:
                task_output = raw_outputs[output_start:output_end]
            else:
                raise ValueError(f"Invalid output dimension: {raw_outputs.ndim}")
            task_probs = F.softmax(task_output, dim=-1)  # Apply softmax
            outputs.append(task_probs)

        return torch.stack(outputs, dim=-2)


def train(
    tasks=["mnist", "fashionmnist", "kmnist", "notmnist"],
    p_active=0.5,
    hidden_size=512,
    num_epochs=5,
    batch_size=128,
    learning_rate=0.001,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
            from notmnist import NotMNISTDataset

            dataset = NotMNISTDataset(root="./data/notMNIST_small")
            input_size = 28 * 28
            n_classes = 10
        elif task == "uciletters":
            from uciletters import OpticalLettersDataset

            dataset = OpticalLettersDataset(file_path="./data/letter-recognition.data")
            input_size = 16
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
    dataloader = DataLoader(multi_task_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
        for inputs, target_probs, active_tasks in loop:
            inputs = inputs.to(device)
            target_probs = target_probs.to(device)  # Shape: [batch_size, num_tasks, n_classes]
            outputs = model(inputs)  # List of outputs (probabilities) for each task

            if epoch == 2:
                pass

            loss = 0
            for i in range(len(tasks)):
                task_outputs = outputs[:, i, :]  # Shape: [batch_size, num_classes]
                task_target_probs = target_probs[:, i, :]  # Shape: [batch_size, num_classes]

                # Compute log of predicted probabilities
                task_log_probs = torch.log(task_outputs)

                # Compute cross-entropy loss using log of probabilities and target probabilities
                task_loss = F.cross_entropy(task_log_probs, task_target_probs)
                loss += task_loss

                # Compute accuracy for both active tasks only
                task_active = active_tasks[:, i]
                if task_active.any():
                    task_indices = torch.where(task_active)[0]
                    task_outputs_active = task_outputs[task_indices]
                    task_target_probs_active = task_target_probs[task_indices]
                    preds = task_outputs_active.argmax(dim=1)
                    correct = preds.eq(task_target_probs_active.argmax(dim=1)).sum().item()
                    correct_counts[i] += correct
                    total_counts[i] += task_target_probs_active.size(0)

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

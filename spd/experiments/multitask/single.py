from collections.abc import Callable
from pathlib import Path
from typing import Any

import einops
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST, KMNIST, MNIST, FashionMNIST, VisionDataset
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), lambda x: x.view(-1, 28**2)]
)
# The to image and ToTensor bit is equivalent to img/255.


class GenericMNISTModel(nn.Module):
    def __init__(self, hidden_size: int, input_size: int = 28 * 28, num_classes: int = 10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.act = nn.ReLU()

    def forward(
        self, x: Float[torch.Tensor, "batch input_size"]
    ) -> Float[torch.Tensor, "batch num_classes"]:
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        return out


class E10MNIST(Dataset[tuple[Tensor, int]]):
    def __init__(
        self, root: str, transform: Callable[[Tensor], Tensor], download: bool, train: bool
    ):
        emnist = EMNIST(root=root, split="letters", download=download, train=train)
        self.transform = transform
        a_to_j = torch.arange(1, 11)
        self.classes = a_to_j - 1
        targets = emnist.targets
        emnist_indices = [i for i, t in enumerate(targets) if t in a_to_j]
        self.data = [emnist.data[i] for i in emnist_indices]
        self.targets = [targets[i] - 1 for i in emnist_indices]
        self.n_classes = 10

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        img = self.transform(img)
        return img, target


class MNISTDropoutDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, dataset: VisionDataset, p: float = 0.25):
        """Output a data with probability p, otherwise output a blank input."""
        self.dataset = dataset
        self.n_classes = len(dataset.classes)
        self.p = p

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        output_blank = torch.rand(1) > self.p
        input, target = self.dataset[index]
        target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=self.n_classes)
        if output_blank:
            input = torch.zeros_like(input)
            target = torch.ones_like(target) / self.n_classes
        return input, target


class MultiMNISTDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, datasets: list[VisionDataset], p: float = 0.25):
        self.datasets = datasets
        self.n_inputs = [d[0][0].shape[-1] for d in datasets]
        self.n_classes = [len(d.classes) for d in datasets]
        self.n_datasets = len(datasets)
        self.lens = [len(d) for d in datasets]
        self.min_length = min(self.lens)
        self.p = p

    def __len__(self) -> int:
        return self.min_length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # mask with probability p
        mask = torch.rand(self.n_datasets) < self.p
        inputs = []
        targets = []
        for i, dataset in enumerate(self.datasets):
            if mask[i]:
                input, target = dataset[index]
                input = input.squeeze(0)
                target = torch.tensor(target)
                target = torch.nn.functional.one_hot(target, num_classes=self.n_classes[i])
            else:
                input = torch.zeros(self.n_inputs[i])
                target = torch.ones(self.n_classes[i]) / self.n_classes[i]
            # print(f"Getting item w/ mask={mask[i]}, input.shape={input.shape}, target={target}")
            targets.append(target)
            inputs.append(input)
        return torch.cat(inputs, dim=-1), torch.cat(targets, dim=-1)


def per_task_softmax(
    pred_logits: Float[torch.Tensor, "... (tasks classes)"],
    n_tasks: int,
    n_classes: int,
) -> Float[torch.Tensor, "... (tasks classes)"]:
    pred_logits = einops.rearrange(
        pred_logits,
        "... (tasks classes) -> ... tasks classes",
        tasks=n_tasks,
        classes=n_classes,
    )
    pred_probs = pred_logits.softmax(dim=-1)
    pred_probs = einops.rearrange(
        pred_probs,
        "... tasks classes -> ... (tasks classes)",
        tasks=n_tasks,
        classes=n_classes,
    )
    return pred_probs


class MultiMNISTDatasetLoss:
    def __init__(self, n_tasks: int, n_classes: int):
        self.n_classes = n_classes
        self.n_tasks = n_tasks
        self.total_output_size = n_tasks * n_classes

    def __call__(
        self,
        pred_logits: Float[torch.Tensor, "... total_output_size"],
        target_probs: Float[torch.Tensor, "... total_output_size"],
    ) -> Float[torch.Tensor, ""]:
        assert pred_logits.shape[-1] == target_probs.shape[-1]
        assert pred_logits.shape[-1] == self.total_output_size
        pred_probs = per_task_softmax(pred_logits, n_tasks=self.n_tasks, n_classes=self.n_classes)
        # We want to sum over classes (normal cross entropy) and then sum over tasks (so that the
        # gradients to individual subnetworks are sensible). Take mean over batch as usual.
        loss = -torch.sum(target_probs * torch.log(pred_probs), dim=-1)
        return loss.mean(dim=0)


def get_data_loaders(
    dataset_class: type[MNIST | KMNIST | FashionMNIST | E10MNIST],
    blanks: bool = False,
    batch_size: int = 128,
    split_ratio: float = 0.8,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    data_dir = "/data/apollo/torch_datasets/"
    # Data transformations
    # Load the dataset and splits
    kwargs = {"root": data_dir, "download": True, "transform": transform}
    train_dataset = dataset_class(train=True, **kwargs)
    test_dataset = dataset_class(train=False, **kwargs)
    if blanks:
        train_dataset = MNISTDropoutDataset(train_dataset)
        test_dataset = MNISTDropoutDataset(test_dataset)
    train_size: int = int(split_ratio * len(train_dataset))
    val_size: int = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(
    dataset_class: type[MNIST | KMNIST | FashionMNIST | E10MNIST],
    blanks: bool = False,
    output_dir: str | None = None,
    input_size: int = 28 * 28,  # MNIST images all are 28x28
    num_classes: int = 10,  # these datasets all have 10 classes
    hidden_size: int = 512,
    batch_size: int = 128,
    num_epochs: int = 3,
    learning_rate: float = 0.001,
    seed: int = 0,
    log_interval: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_class=dataset_class, batch_size=batch_size, blanks=blanks
    )

    # Initialize model, loss function, optimizer
    model: GenericMNISTModel = GenericMNISTModel(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    ).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss: float = 0
        correct: int = 0
        total: int = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]")
        for images, labels in loop:
            images: Float[torch.Tensor, "batch input_size"] = images.view(-1, input_size).to(
                device
            )  # Flatten the 28x28 images
            labels: Float[torch.Tensor, "batch num_classes"] = labels.to(device)
            # Forward pass
            logits: Float[torch.Tensor, "batch num_classes"] = model(images)
            loss: torch.Tensor = criterion(logits, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            total_loss += loss.item()
            _, predicted = logits.max(1)
            if blanks:
                assert logits.shape == labels.shape
                assert logits.shape[-1] == num_classes
                assert logits.ndim == 2
                _, label_int = labels.max(1)
                mask = labels[:, 0] != 1 / num_classes
                total += mask.sum().item()
                correct += predicted.eq(label_int)[mask].sum().item()
            else:
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

        avg_loss: float = total_loss / len(train_loader)
        accuracy: float = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss: float = 0
        val_correct: int = 0
        val_total: int = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images: Float[torch.Tensor, "batch input_size"] = images.view(-1, input_size).to(
                    device
                )
                labels: Float[torch.Tensor, "batch num_classes"] = labels.to(device)
                logits: Float[torch.Tensor, "batch num_classes"] = model(images)
                loss: torch.Tensor = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                if blanks:
                    assert logits.shape == labels.shape
                    assert logits.shape[-1] == num_classes
                    assert logits.ndim == 2
                    _, label_int = labels.max(1)
                    mask = labels[:, 0] != 1 / num_classes
                    val_total += mask.sum().item()
                    val_correct += predicted.eq(label_int)[mask].sum().item()
                else:
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy: float = 100.0 * val_correct / val_total

        if epoch % log_interval == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], "
                f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

    # Testing on the test dataset
    model.eval()
    test_loss: float = 0
    test_correct: int = 0
    test_total: int = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images: Float[torch.Tensor, "batch input_size"] = images.view(-1, input_size).to(device)
            labels: Float[torch.Tensor, "batch num_classes"] = labels.to(device)
            logits: Float[torch.Tensor, "batch num_classes"] = model(images)
            loss: torch.Tensor = criterion(logits, labels)
            test_loss += loss.item()
            _, predicted = logits.max(1)
            if blanks:
                assert logits.shape == labels.shape
                assert logits.shape[-1] == num_classes
                assert logits.ndim == 2
                _, label_int = labels.max(1)
                mask = labels[:, 0] != 1 / num_classes
                test_total += mask.sum().item()
                test_correct += predicted.eq(label_int)[mask].sum().item()
            else:
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy: float = 100.0 * test_correct / test_total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save the model checkpoint
    Path(output_dir).mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path(output_dir) / f"{dataset_class.__name__}.pth")


def main(dataset_class: str, blanks: bool = False):
    dataset_classes = {
        "mnist": MNIST,
        "kmnist": KMNIST,
        "fashion": FashionMNIST,
        "e10mnist": E10MNIST,
    }
    train(dataset_classes[dataset_class], output_dir=f"models/{dataset_class}", blanks=blanks)


if __name__ == "__main__":
    fire.Fire(main)

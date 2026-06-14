import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numpy_to_tensor(
    x: np.ndarray, is_image: bool = True, in_channels: int = 3
) -> torch.Tensor:
    if is_image:
        x = x.astype(np.float32) / 255.0
        if in_channels == 3:
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        elif in_channels == 1:
            mean = np.array([0.1307], dtype=np.float32)
            std = np.array([0.3081], dtype=np.float32)
        else:
            mean = np.array([0.5] * in_channels, dtype=np.float32)
            std = np.array([0.5] * in_channels, dtype=np.float32)
        if x.ndim == 4 and x.shape[-1] == in_channels:
            # Channels last format: (batch, height, width, channels)
            x = (x - mean).astype(np.float32)
            x = x / std
            x = torch.from_numpy(x).permute(0, 3, 1, 2)
        elif x.ndim == 4 and x.shape[1] == in_channels:
            # Channels first format: (batch, channels, height, width)
            mean = mean.reshape(1, in_channels, 1, 1)
            std = std.reshape(1, in_channels, 1, 1)
            x = (x - mean).astype(np.float32)
            x = x / std
            x = torch.from_numpy(x)
        else:
            raise ValueError(
                f"Unexpected image shape: {x.shape}, expected channels={in_channels}"
            )
    else:
        x = torch.from_numpy(x.astype(np.float32))
    return x


def train_and_evaluate(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    train_idx: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 10,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: str | None = None,
    in_channels: int = 3,
    seed: int = 42,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    X_train = X_train_all[train_idx]
    y_train = y_train_all[train_idx]

    X_train_tensor = numpy_to_tensor(X_train, is_image=True, in_channels=in_channels)
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = numpy_to_tensor(X_test, is_image=True, in_channels=in_channels)
    y_test_tensor = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_acc = correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Train Acc: {train_acc:.4f}"
        )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_dataset import load_cifar10_dataset, load_mnist_dataset

    dataset = "cifar10-embedding"

    if dataset == "cifar10-embedding":
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_train_img,
            X_val_img,
            X_test_img,
        ) = load_cifar10_dataset(train_size=1000, test_size=500, random_state=42)
        in_channels = 3
    elif dataset == "mnist-embedding":
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_train_img,
            X_val_img,
            X_test_img,
        ) = load_mnist_dataset(
            dataset_path=None, train_size=1000, test_size=500, random_state=42
        )
        in_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    y_train_all = np.concatenate([y_train, y_val])
    X_train_img_all = np.concatenate([X_train_img, X_val_img])

    np.random.seed(42)
    train_idx = np.random.choice(
        len(X_train_img_all), size=min(500, len(X_train_img_all)), replace=False
    )

    accuracy = train_and_evaluate(
        X_train_img_all,
        y_train_all,
        train_idx,
        X_test_img,
        y_test,
        epochs=5,
        in_channels=in_channels,
    )

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
import os
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange
from unittest.mock import Mock
from typing import Tuple


def compute_accuracy(preds, targets):
    if preds.shape != targets.shape:
        raise RuntimeError("Shapes of predictions and targets must match")
    if len(preds) == 0:
        return torch.tensor(float("nan"))
    result = (targets == preds).float().mean()
    return result


def create_datasets(
    train_root="CIFAR10/train",
    test_root="CIFAR10/test",
    download_train=False,
    download_test=False,
):
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    train_dataset = CIFAR10(
        root=train_root,
        train=True,
        transform=transform,
        download=download_train,
    )
    test_dataset = CIFAR10(
        root=test_root,
        train=False,
        transform=transform,
        download=download_test,
    )
    return train_dataset, test_dataset


def create_model(device, config) -> nn.Module:
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
    model = resnet18(
        pretrained=False,
        num_classes=10,
        zero_init_residual=config["zero_init_residual"],
    )
    model.to(device)
    if os.environ.get("TEST_MODE"):
        wandb.watch = Mock()
    return model


def validate(model, test_loader, device) -> float:
    if not test_loader:
        raise ValueError("Empty test loader provided")
    all_preds = []
    all_labels = []

    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        with torch.inference_mode():
            outputs = model(test_images)
            preds = torch.argmax(outputs, 1)

            all_preds.append(preds)
            all_labels.append(test_labels)
    accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))
    return accuracy


def main(config=None):
    if config is None:
        from hparams import config

    if os.environ.get("TEST_MODE"):
        wandb.init = Mock()
        wandb.log = Mock()
    else:
        wandb.init(config=config, project="effdl_example", name="baseline")

    train_dataset, test_dataset = create_datasets()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device, config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    for epoch in trange(config["epochs"]):
        total_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                accuracy = validate(model, test_loader, device)
                metrics = {"test_acc": accuracy, "train_loss": loss}
                wandb.log(
                    metrics,
                    step=epoch * len(train_dataset) + (i + 1) * config["batch_size"],
                )

        # Логирование после каждой эпохи
        accuracy = validate(model, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        metrics = {"epoch_test_acc": accuracy, "epoch_train_loss": avg_loss}
        wandb.log(metrics, step=(epoch + 1) * len(train_dataset))

    torch.save(model.state_dict(), "model.pt")

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)


if __name__ == "__main__":
    main()

import pytest
import torch
import tempfile
import os
from unittest.mock import patch
from torch.utils.data import DataLoader
from torchvision import transforms
from unittest.mock import Mock
from torchvision.datasets import CIFAR10

import train
from hparams import config


@pytest.fixture
def config():
    return {
        "batch_size": 32,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "epochs": 1,
        "zero_init_residual": False,
    }


@pytest.fixture
def train_dataset():
    # note: реализуйте и протестируйте подготовку данных (скачиание и препроцессинг)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # transforms.Resize((224, 224)),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = CIFAR10(root=tmp_dir, train=True, download=True, transform=transform)
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset, config):
    # note: реализуйте и протестируйте один шаг обучения вместе с метрикой
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available...")

    model = train.create_model(torch.device(device), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    images, labels = next(iter(loader))

    # шаг
    model.eval()
    with torch.inference_mode():
        initial_outputs = model(images.to(device))
        initial_loss = criterion(initial_outputs, labels.to(device))

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        new_outputs = model(images.to(device))
        new_loss = criterion(new_outputs, labels.to(device))

    assert (
        new_loss < initial_loss
    ), f"Loss didn't decrease: {new_loss} >= {initial_loss}"


def test_training(config):
    os.environ["TEST_MODE"] = "1"

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("train.CIFAR10") as mock_cifar, patch(
            "wandb.init", new=Mock()
        ), patch("wandb.log", new=Mock()):

            mock_cifar.side_effect = lambda *args, **kwargs: CIFAR10(
                root=tmp_dir,
                download=True,
                train="train" in kwargs and kwargs["train"],
                transform=transforms.ToTensor(),
            )

            train.main(config=config)

    # Проверяем артефакты
    assert os.path.exists("model.pt"), "Модель не сохранилась"
    os.remove("model.pt")

    # Проверяем логирование
    assert train.wandb.log.call_count >= config["epochs"], "Недостаточно логов"
    assert train.wandb.init.called, "Wandb не инициализирован"

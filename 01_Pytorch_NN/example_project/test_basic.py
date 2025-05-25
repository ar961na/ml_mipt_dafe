import torch
import torch.nn as nn
import pytest
from unittest.mock import patch
from unittest.mock import Mock
from torchvision.models import resnet18

import train


def test_arange_elems():
    arr = torch.arange(0, 10, dtype=torch.float)  # float: 9->9.0
    assert torch.allclose(arr[-1], torch.tensor([9.0]))


def test_div_zero():
    a = torch.zeros(1, dtype=torch.float)  # long->float
    b = torch.ones(1, dtype=torch.float)
    result = b / a
    assert not torch.isfinite(result).any()  # проверка на non-finite


def test_div_zero_python():
    with pytest.raises(ZeroDivisionError):
        1 / 0


def test_accuracy():
    preds = torch.randint(0, 2, size=(100,))
    targets = preds.clone()

    assert train.compute_accuracy(preds, targets) == 1.0

    preds = torch.tensor([1, 2, 3, 0, 0, 0])
    targets = torch.tensor([1, 2, 3, 4, 5, 6])

    assert train.compute_accuracy(preds, targets) == 0.5


@pytest.mark.parametrize(
    "preds,targets,result",
    [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 1.0),
        (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 0]), 0.0),
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 0]), 2 / 3),  # 2/5->2/3 поправила
    ],
)
def test_accuracy_parametrized(preds, targets, result):
    assert torch.allclose(
        train.compute_accuracy(preds, targets),
        torch.tensor([result]),
        rtol=0,
        atol=1e-5,
    )


def test_compute_accuracy_edge_cases():
    # Все предсказания неверны
    preds = torch.tensor([1, 2, 3])
    targets = torch.tensor([4, 5, 6])
    assert train.compute_accuracy(preds, targets) == 0.0

    # Пустые тензоры
    preds = torch.empty(0, dtype=torch.float)
    targets = torch.empty(0, dtype=torch.float)
    assert torch.isnan(train.compute_accuracy(preds, targets))

    # Разная длина тензоров
    with pytest.raises(RuntimeError):
        train.compute_accuracy(torch.tensor([0]), torch.tensor([0, 1]))


@pytest.mark.parametrize("input_shape", [(10,), (10, 1), (10, 5)])
def test_compute_accuracy_shapes(input_shape):
    preds = torch.randint(0, 10, input_shape)
    targets = torch.randint(0, 10, input_shape)
    acc = train.compute_accuracy(preds, targets)
    assert 0.0 <= acc <= 1.0


def test_create_datasets():
    train_ds, test_ds = train.create_datasets()

    assert len(train_ds) == 50000, "Неправильный размер тренировочного набора"
    assert len(test_ds) == 10000, "Неправильный размер тестового набора"

    img, label = train_ds[0]
    assert img.shape == (3, 32, 32), "Неправильная форма изображений"
    assert 0 <= label <= 9, "Некорректная метка"


def test_validate_function():
    model = resnet18(weights=None, num_classes=10)
    device = torch.device("cpu")
    model.to(device)

    dummy_data = (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))
    loader = [dummy_data] * 3

    accuracy = train.validate(model, loader, device)
    assert 0.0 <= accuracy <= 1.0


def test_validation_with_empty_data():
    model = resnet18(pretrained=False, num_classes=10)
    device = torch.device("cpu")
    empty_loader = []

    with pytest.raises(ValueError):
        train.validate(model, empty_loader, device)

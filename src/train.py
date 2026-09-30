"""Обучение классификатора комнат: baseline (только fc) -> fine-tuning (layer4 + fc)."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import (
    BASELINE_WEIGHTS,
    CLASSES,
    FINETUNED_WEIGHTS,
    HISTORY_PATH,
    METRICS_PATH,
    PROCESSED_DATA_DIR,
    SEED,
)
from src.model import build_classifier
from src.transforms import eval_transforms, train_transforms
from src.utils import get_device, set_seed


def freeze_frozen_batchnorm(model: nn.Module) -> None:
    """requires_grad=False не останавливает обновление running mean/var в BatchNorm,
    поэтому BN-слои замороженной части сети нужно явно держать в eval()."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    freeze_frozen_batchnorm(model)
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        running_loss += criterion(outputs, labels).item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs: int, save_path: Path) -> dict:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc, current_lr)):
            history[key].append(value)

        log = (f"Epoch {epoch + 1:02d}/{num_epochs:02d} [{time.time() - start:.0f}s] | "
               f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.1f}% | "
               f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.1f}% | LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc, "classes": CLASSES}, save_path)
            log += " -> [Saved]"
        print(log)
    return history


@torch.inference_mode()
def evaluate(model, weights_path: Path, loader, device) -> dict:
    model.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    for inputs, labels in loader:
        y_pred.extend(model(inputs.to(device)).argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True),
    }


def make_loaders(batch_size: int, num_workers: int):
    train_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "train", transform=train_transforms)
    val_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "val", transform=eval_transforms)
    test_ds = datasets.ImageFolder(PROCESSED_DATA_DIR / "test", transform=eval_transforms)
    assert train_ds.classes == CLASSES, f"Классы в данных {train_ds.classes} не совпадают с конфигом {CLASSES}"

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-baseline", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_loaders(args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()

    # --- 1. Baseline: обучаем только голову ---
    model = build_classifier(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)

    print("\n=== Baseline (frozen backbone) ===")
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    baseline_history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                   device, args.epochs_baseline, BASELINE_WEIGHTS)
    baseline_metrics = evaluate(model, BASELINE_WEIGHTS, test_loader, device)

    # --- 2. Fine-tuning: размораживаем layer4, стартуем с лучшего baseline ---
    for param in model.layer4.parameters():
        param.requires_grad = True

    print("\n=== Fine-tuning (layer4 + fc) ===")
    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    finetuned_history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                    device, args.epochs_finetune, FINETUNED_WEIGHTS)
    finetuned_metrics = evaluate(model, FINETUNED_WEIGHTS, test_loader, device)

    metrics = {"baseline": baseline_metrics, "finetuned": finetuned_metrics}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    history = {"baseline": baseline_history, "finetuned": finetuned_history}
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n=== Test ===")
    print(f"{'Модель':<22} | {'Accuracy':>8} | {'Macro-F1':>8}")
    for name, m in metrics.items():
        print(f"{name:<22} | {m['accuracy'] * 100:7.2f}% | {m['macro_f1']:8.4f}")
    print(f"\nМетрики сохранены в {METRICS_PATH}")


if __name__ == "__main__":
    main()

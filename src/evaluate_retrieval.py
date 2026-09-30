"""Оценка качества поиска: база = train, запросы = test, релевантен объект того же типа комнаты."""
import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import FINETUNED_WEIGHTS, PROCESSED_DATA_DIR, RETRIEVAL_METRICS_PATH
from src.model import build_classifier, extract_embeddings, get_embedding_extractor, load_classifier
from src.transforms import eval_transforms
from src.utils import get_device

KS = (1, 5, 10)


@dataclass
class Encoder:
    name: str
    model: nn.Module
    transform: Callable


class ClipImageEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(x)


def resnet_imagenet_encoder(device: torch.device) -> Encoder:
    model = build_classifier(pretrained=True).to(device)
    return Encoder("ResNet18 (ImageNet)", get_embedding_extractor(model), eval_transforms)


def resnet_finetuned_encoder(device: torch.device) -> Encoder:
    model = load_classifier(FINETUNED_WEIGHTS, device)
    return Encoder("ResNet18 (fine-tuned)", get_embedding_extractor(model), eval_transforms)


def clip_encoder(device: torch.device) -> Encoder:
    import open_clip  # опциональная зависимость, см. requirements-dev.txt

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    return Encoder("CLIP ViT-B/32 (zero-shot)", ClipImageEncoder(model.to(device)).eval(), preprocess)


def encode_split(encoder: Encoder, split: str, device: torch.device, batch_size: int = 32):
    """Возвращает (эмбеддинги, метки, пути) для train/val/test."""
    dataset = datasets.ImageFolder(PROCESSED_DATA_DIR / split, transform=encoder.transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = np.vstack([extract_embeddings(encoder.model, x.to(device)) for x, _ in loader])
    labels = np.array(dataset.targets)
    paths = [p for p, _ in dataset.samples]
    return embeddings, labels, paths


def retrieval_metrics(query_emb, query_labels, db_emb, db_labels, ks=KS) -> dict:
    """Precision@K и mAP@K. Эмбеддинги L2-нормированы, поэтому q @ db.T = косинусное сходство."""
    max_k = max(ks)
    top = np.argsort(-(query_emb @ db_emb.T), axis=1)[:, :max_k]
    relevant = (db_labels[top] == query_labels[:, None]).astype(float)

    metrics = {}
    for k in ks:
        rel_k = relevant[:, :k]
        precision_at_i = np.cumsum(rel_k, axis=1) / np.arange(1, k + 1)
        ap = (precision_at_i * rel_k).sum(axis=1) / k
        metrics[f"P@{k}"] = float(rel_k.mean())
        metrics[f"mAP@{k}"] = float(ap.mean())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clip", action="store_true", help="Не оценивать CLIP (не нужен open_clip)")
    args = parser.parse_args()

    device = get_device()
    builders = [resnet_imagenet_encoder, resnet_finetuned_encoder]
    if not args.no_clip:
        builders.append(clip_encoder)

    results = {}
    for build in builders:
        encoder = build(device)
        print(f"Считаем эмбеддинги: {encoder.name}...")
        db_emb, db_labels, _ = encode_split(encoder, "train", device)
        q_emb, q_labels, _ = encode_split(encoder, "test", device)
        results[encoder.name] = retrieval_metrics(q_emb, q_labels, db_emb, db_labels)

    RETRIEVAL_METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    columns = list(next(iter(results.values())))
    print(f"\n{'Энкодер':<28} | " + " | ".join(f"{c:>7}" for c in columns))
    for name, m in results.items():
        print(f"{name:<28} | " + " | ".join(f"{m[c]:7.4f}" for c in columns))
    print(f"\nМетрики сохранены в {RETRIEVAL_METRICS_PATH}")


if __name__ == "__main__":
    main()

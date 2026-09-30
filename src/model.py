from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.config import NUM_CLASSES


def build_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(weights_path: Path, device: torch.device) -> nn.Module:
    model = build_classifier(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def get_embedding_extractor(model: nn.Module) -> nn.Module:
    """ResNet без последнего fc-слоя: на выходе вектор (N, 512, 1, 1)."""
    return nn.Sequential(*list(model.children())[:-1]).eval()


@torch.inference_mode()
def extract_embeddings(extractor: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    """L2-нормированные эмбеддинги: скалярное произведение в FAISS = косинусное сходство."""
    embeddings = extractor(inputs).flatten(1).cpu().numpy()
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype("float32")

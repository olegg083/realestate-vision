import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from src.config import CLASSES, FAISS_INDEX_PATH, FINETUNED_WEIGHTS, METADATA_PATH
from src.model import get_embedding_extractor, load_classifier
from src.transforms import eval_transforms
from src.utils import load_faiss_index

ApartmentFilter = Callable[[dict], bool]


@dataclass
class RoomPrediction:
    room_type: str
    confidence: float


@dataclass
class SearchHit:
    similarity: float
    apartment: dict


class SearchEngine:
    """Классификатор типа комнаты + поиск ближайших объектов в FAISS за один проход backbone."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        classifier = load_classifier(FINETUNED_WEIGHTS, self.device)
        self.backbone = get_embedding_extractor(classifier)
        self.head = classifier.fc.eval()

        self.index = load_faiss_index(FAISS_INDEX_PATH)
        self.metadata: list[dict] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"Индекс ({self.index.ntotal}) и метаданные ({len(self.metadata)}) не согласованы — "
                "пересоберите их командой python -m src.build_index"
            )

    @property
    def size(self) -> int:
        return self.index.ntotal

    @torch.inference_mode()
    def search(self, image: Image.Image, k: int,
               apartment_filter: ApartmentFilter | None = None) -> tuple[RoomPrediction, list[SearchHit]]:
        x = eval_transforms(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.backbone(x).flatten(1)

        probs = torch.softmax(self.head(features), dim=1)[0]
        top_class = int(probs.argmax())
        prediction = RoomPrediction(CLASSES[top_class], float(probs[top_class]))

        embedding = features.cpu().numpy()
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)

        # IndexFlatIP не умеет фильтровать по метаданным, поэтому при фильтрах ищем по всей базе
        n_candidates = self.size if apartment_filter else min(k, self.size)
        similarities, indices = self.index.search(embedding.astype("float32"), n_candidates)

        hits = []
        for similarity, idx in zip(similarities[0], indices[0], strict=True):
            if idx < 0:
                continue
            apartment = self.metadata[idx]
            if apartment_filter and not apartment_filter(apartment):
                continue
            hits.append(SearchHit(float(similarity), apartment))
            if len(hits) == k:
                break
        return prediction, hits

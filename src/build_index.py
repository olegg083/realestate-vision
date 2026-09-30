"""Извлечение эмбеддингов базы объектов и построение FAISS-индекса с метаданными."""
import argparse
import json
import random
from pathlib import Path

import faiss
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import (
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    FINETUNED_WEIGHTS,
    METADATA_PATH,
    PROCESSED_DATA_DIR,
    SEED,
)
from src.model import extract_embeddings, get_embedding_extractor, load_classifier
from src.transforms import eval_transforms
from src.utils import get_device, save_faiss_index


def generate_metadata(image_paths: list[Path], room_types: list[str]) -> list[dict]:
    """Синтетические объявления: в датасете нет реальных цен и площадей."""
    rng = random.Random(SEED)
    return [
        {
            "apartment_id": f"APT-{idx:05d}",
            "image_path": path.relative_to(PROCESSED_DATA_DIR).as_posix(),
            "room_type": room_type,
            "price_mln": round(rng.uniform(5.0, 35.0), 1),
            "area_sqm": rng.randint(30, 120),
            "rooms": rng.choice([1, 2, 3, 4]),
        }
        for idx, (path, room_type) in enumerate(zip(image_paths, room_types))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="Какую часть данных положить в базу поиска")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    extractor = get_embedding_extractor(load_classifier(FINETUNED_WEIGHTS, device))

    dataset = datasets.ImageFolder(PROCESSED_DATA_DIR / args.split, transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Извлекаем эмбеддинги для {len(dataset)} изображений...")
    embeddings = np.vstack([extract_embeddings(extractor, inputs.to(device)) for inputs, _ in loader])

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_faiss_index(index, FAISS_INDEX_PATH)
    print(f"FAISS индекс ({index.ntotal} векторов) сохранён в {FAISS_INDEX_PATH}")

    image_paths = [Path(p) for p, _ in dataset.samples]
    room_types = [dataset.classes[label] for _, label in dataset.samples]
    metadata = generate_metadata(image_paths, room_types)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Метаданные сохранены в {METADATA_PATH}")


if __name__ == "__main__":
    main()

"""Разбиение MIT Indoor Scenes на train/val/test для целевых классов комнат."""
import random
import shutil

from src.config import CLASSES, PROCESSED_DATA_DIR, RAW_DATA_DIR, SEED, TRAIN_RATIO, VAL_RATIO


def prepare_dataset() -> None:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Не найден исходный датасет: {RAW_DATA_DIR}\n"
            "Скачайте MIT Indoor Scenes и распакуйте в data/raw/ (см. README)."
        )

    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    rng = random.Random(SEED)

    for class_name in CLASSES:
        images = sorted((RAW_DATA_DIR / class_name).glob("*.jpg"))
        rng.shuffle(images)

        train_end = int(len(images) * TRAIN_RATIO)
        val_end = train_end + int(len(images) * VAL_RATIO)
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, split_images in splits.items():
            target_dir = PROCESSED_DATA_DIR / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy(img, target_dir / img.name)

        print(f"{class_name:12s} | " + " | ".join(f"{s}: {len(v):3d}" for s, v in splits.items()))


if __name__ == "__main__":
    prepare_dataset()
    print(f"Датасет подготовлен: {PROCESSED_DATA_DIR}")

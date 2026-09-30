import random
from pathlib import Path

import faiss
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# faiss.read_index/write_index используют fopen и не открывают пути с кириллицей на Windows
def save_faiss_index(index: faiss.Index, path: Path) -> None:
    path.write_bytes(faiss.serialize_index(index).tobytes())


def load_faiss_index(path: Path) -> faiss.Index:
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))

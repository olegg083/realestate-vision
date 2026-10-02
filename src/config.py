from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw" / "indoorCVPR_09" / "Images"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

BASELINE_WEIGHTS = MODELS_DIR / "best_baseline.pth"
FINETUNED_WEIGHTS = MODELS_DIR / "best_finetuned.pth"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
HISTORY_PATH = MODELS_DIR / "history.json"
RETRIEVAL_METRICS_PATH = MODELS_DIR / "retrieval_metrics.json"

GITHUB_REPO = "olegg083/realestate-vision"
MODELS_RELEASE_TAG = "v1"
RELEASE_ARTIFACTS = [FINETUNED_WEIGHTS, FAISS_INDEX_PATH, METADATA_PATH,
                     METRICS_PATH, HISTORY_PATH, RETRIEVAL_METRICS_PATH]

CLASSES = ["bathroom", "bedroom", "dining_room", "kitchen", "livingroom"]
NUM_CLASSES = len(CLASSES)
EMBEDDING_DIM = 512

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

SEED = 42

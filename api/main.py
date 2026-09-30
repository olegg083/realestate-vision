import io
import json

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.config import FAISS_INDEX_PATH, FINETUNED_WEIGHTS, METADATA_PATH
from src.model import extract_embeddings, get_embedding_extractor, load_classifier
from src.transforms import eval_transforms
from src.utils import load_faiss_index

app = FastAPI(title="RealEstate Vision API")
device = torch.device("cpu")

print("Загрузка модели...")
extractor = get_embedding_extractor(load_classifier(FINETUNED_WEIGHTS, device))

print("Загрузка FAISS индекса...")
index = load_faiss_index(FAISS_INDEX_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

print("API готово к работе")


@app.post("/search")
async def search_apartments(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = eval_transforms(image).unsqueeze(0).to(device)
        emb = extract_embeddings(extractor, input_tensor)

        k = 5
        distances, indices = index.search(emb, k)

        results = [
            {"similarity_score": round(float(sim), 3), "apartment": metadata[int(idx)]}
            for sim, idx in zip(distances[0], indices[0])
        ]
        return JSONResponse(content={"status": "success", "results": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

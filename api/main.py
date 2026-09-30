from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
import faiss
import json
import numpy as np
from PIL import Image
import io

app = FastAPI(title="RealEstate Vision API")
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загружаем модель-экстрактор
print("Загрузка модели...")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 5)
checkpoint = torch.load('models/best_finetuned.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
extractor = nn.Sequential(*list(model.children())[:-1]).eval().to(device)

# Загружаем FAISS и метаданные
print("Загрузка FAISS индекса...")
index = faiss.read_index('models/faiss_index.bin')

with open('models/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print("API готово к работе")

# --- 2. ЭНДПОИНТ ДЛЯ ПОИСКА ---
@app.post("/search")
async def search_apartments(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = extractor(input_tensor).view(1, -1).numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            
        k = 5
        distances, indices = index.search(emb.astype('float32'), k)
        
        results = []
        for i in range(k):
            idx = str(indices[0][i])
            sim = float(distances[0][i])
            
            apt_info = metadata[idx]
            results.append({
                "similarity_score": round(sim, 3),
                "apartment": apt_info
            })
            
        return JSONResponse(content={"status": "success", "results": results})
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.config import CLASSES, PROCESSED_DATA_DIR

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("realestate_vision.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # В тестах движок подменяется заранее, тогда модель и индекс не загружаются
    if getattr(app.state, "engine", None) is None:
        from src.search import SearchEngine

        logger.info("Загрузка модели и FAISS индекса...")
        app.state.engine = SearchEngine()
    logger.info("API готово, объектов в индексе: %d", app.state.engine.size)
    yield


app = FastAPI(
    title="RealEstate Vision API",
    description="Визуальный поиск похожих квартир по фотографии комнаты",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/images", StaticFiles(directory=PROCESSED_DATA_DIR, check_dir=False), name="images")


class Apartment(BaseModel):
    apartment_id: str
    image_path: str
    room_type: str
    price_mln: float
    area_sqm: int
    rooms: int


class SearchHit(BaseModel):
    similarity_score: float
    image_url: str
    apartment: Apartment


class RoomPrediction(BaseModel):
    room_type: str
    confidence: float


class SearchResponse(BaseModel):
    query_room: RoomPrediction
    results: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    index_size: int


def read_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Неподдерживаемый тип файла {file.content_type}. "
                                 f"Допустимы: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Файл больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Не удалось прочитать изображение")
    return image


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    return HealthResponse(status="ok", index_size=request.app.state.engine.size)


@app.post("/search", response_model=SearchResponse)
def search(
    request: Request,
    file: UploadFile = File(..., description="Фото комнаты (JPEG, PNG, WebP)"),
    k: int = Query(5, ge=1, le=50, description="Сколько объектов вернуть"),
    room_type: str | None = Query(None, description=f"Фильтр по типу комнаты: {', '.join(CLASSES)}"),
    min_price: float | None = Query(None, ge=0, description="Минимальная цена, млн ₽"),
    max_price: float | None = Query(None, ge=0, description="Максимальная цена, млн ₽"),
    rooms: int | None = Query(None, ge=1, description="Количество комнат"),
):
    if room_type is not None and room_type not in CLASSES:
        raise HTTPException(422, f"Неизвестный room_type '{room_type}'. Допустимы: {', '.join(CLASSES)}")

    image = read_image(file)

    def apartment_filter(apt: dict) -> bool:
        return ((room_type is None or apt["room_type"] == room_type)
                and (min_price is None or apt["price_mln"] >= min_price)
                and (max_price is None or apt["price_mln"] <= max_price)
                and (rooms is None or apt["rooms"] == rooms))

    has_filters = any(v is not None for v in (room_type, min_price, max_price, rooms))
    prediction, hits = request.app.state.engine.search(image, k, apartment_filter if has_filters else None)

    return SearchResponse(
        query_room=RoomPrediction(room_type=prediction.room_type, confidence=round(prediction.confidence, 3)),
        results=[
            SearchHit(
                similarity_score=round(hit.similarity, 3),
                image_url=str(request.url_for("images", path=hit.apartment["image_path"])),
                apartment=Apartment(**hit.apartment),
            )
            for hit in hits
        ],
    )

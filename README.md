# 🏠 RealEstate Vision

Визуальный поиск похожих квартир по фотографии комнаты. Пользователь загружает фото (кухня, спальня, ванная и т.д.), сервис находит в базе объявлений комнаты с наиболее похожим интерьером.

**Стек:** PyTorch · torchvision (ResNet18) · FAISS · FastAPI · Streamlit

## Как это работает

```mermaid
flowchart LR
    A[Фото комнаты] --> B[ResNet18<br/>fine-tuned]
    B --> C[Эмбеддинг 512d<br/>L2-норма]
    C --> D[FAISS<br/>IndexFlatIP]
    D --> E[Top-K похожих<br/>объявлений]
    subgraph Offline
        F[Train-изображения] --> B2[ResNet18] --> G[FAISS индекс<br/>+ метаданные]
    end
```

1. **Классификатор.** ResNet18, предобученный на ImageNet, дообучается классифицировать 5 типов комнат. Это нужно, чтобы эмбеддинги отражали признаки интерьера, а не общие признаки ImageNet.
2. **Эмбеддинги.** С сети снимается последний `fc`-слой, выход global average pooling (512 чисел) L2-нормируется.
3. **Поиск.** Нормированные векторы лежат в `faiss.IndexFlatIP`. Скалярное произведение нормированных векторов равно косинусному сходству.
4. **Сервис.** FastAPI принимает изображение и возвращает top-K объектов с метаданными. Streamlit служит веб-интерфейсом поверх API.

## Данные

[MIT Indoor Scenes (indoorCVPR_09)](https://web.mit.edu/torralba/www/indoor.html). Используются 5 классов: `bathroom`, `bedroom`, `dining_room`, `kitchen`, `livingroom`. Каждый класс разбит на train/val/test в пропорции 70/15/15.

| Класс | Train | Val | Test |
|---|---:|---:|---:|
| bathroom | 137 | 29 | 31 |
| bedroom | 463 | 99 | 100 |
| dining_room | 191 | 41 | 42 |
| kitchen | 513 | 110 | 111 |
| livingroom | 494 | 105 | 107 |

> ⚠️ Цены, площади и число комнат в `metadata.json` **синтетические** (генерируются с фиксированным seed). В исходном датасете есть только изображения.

## Результаты

Классификация типа комнаты на test-выборке (391 изображение):

| Модель | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline: замороженный backbone, обучается только `fc` | 76.98% | 0.761 |
| Fine-tuning: `layer4` + `fc` | **82.61%** | **0.817** |

После запуска `src.train` актуальные метрики сохраняются в `models/metrics.json`.

### Качество поиска

Классификатор обучался на accuracy, но продукт — это поиск, поэтому он оценивается отдельно (`python -m src.evaluate_retrieval`):

- база поиска — train (1798 изображений), запросы — test (391);
- результат релевантен, если тип комнаты совпадает с запросом;
- метрики: Precision@K и mAP@K.

| Энкодер | P@1 | P@5 | P@10 | mAP@10 |
|---|---:|---:|---:|---:|
| ResNet18 (ImageNet, без дообучения) | | | | |
| ResNet18 (fine-tuned) | | | | |
| CLIP ViT-B/32 (zero-shot) | | | | |

Разбор результатов, t-SNE эмбеддингов и примеры выдачи — в [`notebooks/03_embeddings_and_search.ipynb`](notebooks/03_embeddings_and_search.ipynb).

## Ноутбуки

Весь пайплайн запускается скриптами из `src/`. Ноутбуки ничего не обучают: они импортируют код из `src` и анализируют результаты.

| Ноутбук | Содержание |
|---|---|
| [`01_eda`](notebooks/01_eda.ipynb) | Распределение классов, примеры, размеры и форматы изображений |
| [`02_training_analysis`](notebooks/02_training_analysis.ipynb) | Кривые обучения, метрики на test, confusion matrix, разбор ошибок |
| [`03_embeddings_and_search`](notebooks/03_embeddings_and_search.ipynb) | Сравнение энкодеров для поиска, t-SNE, демо выдачи |

## Структура проекта

```
├── src/                 # ML-пайплайн
│   ├── config.py        # пути, классы, гиперпараметры
│   ├── transforms.py    # аугментации и препроцессинг
│   ├── model.py         # ResNet18, загрузка весов, извлечение эмбеддингов
│   ├── prepare_data.py  # разбиение датасета на train/val/test
│   ├── train.py         # baseline -> fine-tuning, оценка на test
│   ├── build_index.py   # эмбеддинги базы + FAISS индекс + метаданные
│   ├── evaluate_retrieval.py  # Precision@K / mAP@K для разных энкодеров
│   └── search.py        # поисковый движок: классификация + FAISS + фильтры
├── api/main.py          # FastAPI: /search, /health, /images
├── tests/               # тесты API
├── app/streamlit_app.py # веб-интерфейс
├── notebooks/           # EDA и анализ результатов
├── data/                # (не в git) raw и processed изображения
└── models/              # (не в git) веса, индекс, метаданные
```

## Запуск

Требуется Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt  # для ноутбуков: requirements-dev.txt
```

**1. Данные.** Скачайте датасет и распакуйте так, чтобы получился путь `data/raw/indoorCVPR_09/Images/<класс>/*.jpg`. Затем:

```bash
python -m src.prepare_data
```

**2. Обучение и индекс.** Все команды выполняются из корня репозитория. Обучение на GPU занимает около 8 минут.

```bash
python -m src.train
python -m src.build_index
python -m src.evaluate_retrieval   # нужен open_clip_torch из requirements-dev.txt, либо флаг --no-clip
```

**3. Сервис.** API и интерфейс запускаются в двух терминалах:

```bash
uvicorn api.main:app --port 8000
streamlit run app/streamlit_app.py
```

Интерфейс откроется на http://localhost:8501, Swagger API доступен на http://localhost:8000/docs.

Адрес API для интерфейса задаётся переменной окружения `API_URL` (по умолчанию `http://127.0.0.1:8000`).

## API

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/search` | Поиск похожих объектов по фото |
| `GET` | `/health` | Статус сервиса и размер индекса |
| `GET` | `/images/{path}` | Фотографии объектов из базы |

Параметры `/search`: `file` — изображение (JPEG, PNG или WebP, до 10 МБ), `k` — число результатов (1–50, по умолчанию 5). Необязательные фильтры: `room_type`, `min_price`, `max_price`, `rooms`.

```bash
curl -X POST "http://localhost:8000/search?k=3&room_type=kitchen&max_price=20" -F "file=@kitchen.jpg;type=image/jpeg"
```

Пример ответа (значения условные):

```json
{
  "query_room": {"room_type": "kitchen", "confidence": 0.94},
  "results": [
    {
      "similarity_score": 0.871,
      "image_url": "http://localhost:8000/images/train/kitchen/int474.jpg",
      "apartment": {"apartment_id": "APT-00812", "image_path": "train/kitchen/int474.jpg",
                    "room_type": "kitchen", "price_mln": 14.2, "area_sqm": 64, "rooms": 2}
    }
  ]
}
```

Кроме поиска соседей, модель возвращает тип комнаты на запросе (`query_room`): голова классификатора работает поверх того же прохода backbone, так что это не требует дополнительных вычислений.

Ошибки возвращаются с понятными кодами: `400` — файл не читается как изображение, `413` — файл слишком большой, `415` — неподдерживаемый формат, `422` — неверные параметры.

## Тесты

```bash
pytest
```

Тесты API подменяют поисковый движок фейковым, поэтому не требуют обученной модели и работают без GPU.

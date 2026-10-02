import io
import os

import requests
import streamlit as st
from PIL import Image, ImageOps

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
THUMBNAIL_SIZE = (360, 270)
QUERY_IMAGE_WIDTH = 400
ROOM_TYPES = {
    "bathroom": "Ванная",
    "bedroom": "Спальня",
    "dining_room": "Столовая",
    "kitchen": "Кухня",
    "livingroom": "Гостиная",
}

st.set_page_config(page_title="RealEstate Vision", page_icon="🏠", layout="wide")


@st.cache_data(show_spinner=False)
def fetch_thumbnail(url: str) -> Image.Image | None:
    """Фото в базе разного размера и пропорций: приводим к одному кадру 4:3, чтобы карточки были одинаковыми."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except (requests.RequestException, OSError):
        return None
    return ImageOps.fit(image, THUMBNAIL_SIZE, Image.Resampling.LANCZOS)


def api_error_message(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) else f"Ошибка сервера: {response.status_code}"


# --- Боковая панель: параметры поиска ---
with st.sidebar:
    st.header("Параметры поиска")
    k = st.slider("Сколько объектов показать", min_value=3, max_value=15, value=6, step=3)
    room_type = st.selectbox("Тип комнаты", [None, *ROOM_TYPES],
                             format_func=lambda r: "Любой" if r is None else ROOM_TYPES[r])
    price_range = st.slider("Цена, млн ₽", min_value=5.0, max_value=35.0, value=(5.0, 35.0), step=0.5)
    rooms = st.selectbox("Комнат в квартире", [None, 1, 2, 3, 4],
                         format_func=lambda r: "Любое" if r is None else str(r))

st.title("🏠 RealEstate Vision")
st.markdown("### Визуальный поиск похожих квартир на основе Deep Learning")

uploaded_file = st.file_uploader("Загрузите фотографию комнаты (кухня, спальня, ванная и т.д.)",
                                 type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.subheader("Ваш запрос:")
    st.image(uploaded_file, width=QUERY_IMAGE_WIDTH)

    params = {"k": k, "min_price": price_range[0], "max_price": price_range[1]}
    if room_type:
        params["room_type"] = room_type
    if rooms:
        params["rooms"] = rooms

    with st.spinner("Нейросеть анализирует изображение и ищет совпадения..."):
        try:
            response = requests.post(
                f"{API_URL}/search",
                params=params,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=30,
            )
        except requests.RequestException as e:
            st.error(f"Не удалось подключиться к API ({API_URL}). Сервер запущен? Ошибка: {e}")
            st.stop()

    if not response.ok:
        st.error(api_error_message(response))
        st.stop()

    data = response.json()
    query_room = data["query_room"]
    st.info(f"Похоже на: **{ROOM_TYPES.get(query_room['room_type'], query_room['room_type'])}** "
            f"(уверенность {query_room['confidence'] * 100:.0f}%)")

    results = data["results"]
    if not results:
        st.warning("Ничего не найдено — попробуйте ослабить фильтры.")
        st.stop()

    st.success(f"Найдено {len(results)} похожих объектов!")
    st.markdown("---")
    st.subheader("Рекомендации из базы недвижимости:")

    cols = st.columns(3)
    for idx, res in enumerate(results):
        apt = res["apartment"]
        with cols[idx % 3]:
            thumbnail = fetch_thumbnail(res["image_url"])
            if thumbnail:
                st.image(thumbnail, width=THUMBNAIL_SIZE[0])
            else:
                st.error("Фото не найдено")

            st.markdown(f"### 💰 {apt['price_mln']} млн ₽")
            st.markdown(f"**Тип:** {ROOM_TYPES.get(apt['room_type'], apt['room_type'])} | "
                        f"**Сходство:** {res['similarity_score'] * 100:.1f}%")
            st.markdown(f"📏 {apt['area_sqm']} м² • 🚪 {apt['rooms']} комн.")
            st.markdown(f"🆔 ID: `{apt['apartment_id']}`")
            st.markdown("---")

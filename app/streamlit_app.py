import streamlit as st
import requests
from PIL import Image
import io
import os

# Настройка страницы
st.set_page_config(page_title="RealEstate Vision", page_icon="🏠", layout="wide")

st.title("🏠 RealEstate Vision")
st.markdown("### Визуальный поиск похожих квартир на основе Deep Learning")

# URL нашего FastAPI сервера
API_URL = "http://127.0.0.1:8000/search"

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите фотографию комнаты (Кухня, Спальня, Ванная и т.д.)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Показываем загруженную картинку
    st.subheader("Ваш запрос:")
    image = Image.open(uploaded_file)
    st.image(image, width=400)

    # 2. Отправляем запрос на наш API
    with st.spinner("Нейросеть анализирует изображение и ищет совпадения в FAISS..."):
        try:
            # Превращаем файл обратно в байты для отправки по HTTP
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    st.success(f"Найдено {len(results)} похожих объектов!")
                    st.markdown("---")
                    st.subheader("Рекомендации из базы недвижимости:")
                    
                    # 3. Рисуем карточки в 3 колонки
                    cols = st.columns(3)
                    
                    for idx, res in enumerate(results):
                        apt = res["apartment"]
                        sim = res["similarity_score"]
                        
                        # Исправляем путь к картинке (убираем ../, так как запускаем из корня)
                        img_path = apt["image_path"].replace("../", "")
                        
                        # Рисуем карточку в нужной колонке
                        with cols[idx % 3]:
                            if os.path.exists(img_path):
                                st.image(Image.open(img_path), use_container_width=True)
                            else:
                                st.error("Фото не найдено на диске")
                                
                            st.markdown(f"### 💰 {apt['price_mln']} млн ₽")
                            st.markdown(f"**Тип:** {apt['room_type'].capitalize()} | **Сходство:** {sim*100:.1f}%")
                            st.markdown(f"📏 {apt['area_sqm']} м² • 🚪 {apt['rooms']} комн.")
                            st.markdown(f"🆔 ID: `{apt['apartment_id']}`")
                            st.markdown("---")
                else:
                    st.warning("Ничего не найдено.")
            else:
                st.error(f"Ошибка сервера: {response.status_code}")
                
        except Exception as e:
            st.error(f"Не удалось подключиться к API. Сервер работает? Ошибка: {e}")
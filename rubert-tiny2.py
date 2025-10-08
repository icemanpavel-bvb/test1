import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Анализ эмоций", page_icon="😊")
st.title("😊 Анализ эмоций русского текста")


@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="cointegrated/rubert-tiny2-cedr-emotion-detection")

with st.spinner("Загружаем модель эмоций..."):
    classifier = load_model()

# Интерфейс
text = st.text_area(
    "Введите текст на русском языке для анализа эмоции:",
    placeholder="вот сюда пиши(те)",
    height=100
)

if st.button("Анализировать", type="primary"):
    if text.strip():
        with st.spinner("Работаем..."):
            result = classifier(text)[0]

        # Отображение результата
        emotion = result['label']
        confidence = result['score']

        st.success("Результат анализа:")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Эмоция", emotion)
        with col2:
            st.metric("🎯 Уверенность", f"{confidence:.1%}")

        st.progress(float(confidence), text=f"Уверенность: {confidence:.1%}")

    else:
        st.warning("Пожалуйста, введите текст для анализа")
import cv2
import streamlit as st
import numpy as np

st.set_page_config(page_title="Procesamiento de Imágenes", layout="centered")

st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 28px;
            color: #4A90E2;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-top: 15px;
            margin-bottom: 10px;
        }
        .stImage img {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 80%;
            height: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🖼️ Procesamiento y Mejora de Imágenes</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png"])
if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equ = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    with col2:
        st.image(equ, caption="Contraste Mejorado", use_container_width=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.image(cl1, caption="CLAHE Aplicado", use_container_width=True)
    with col4:
        st.image(binary, caption="Binarizada", use_container_width=True)

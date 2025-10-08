import cv2
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Editor de Imágenes | Rotación y Escala",
    page_icon="🖼️",
    layout="centered"
)

st.markdown("""
<style>
    body {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Segoe UI', sans-serif;
    }

    .main {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 2rem 3rem;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.08);
        max-width: 850px;
        margin: auto;
    }

    h1 {
        text-align: center;
        color: #334155;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .upload-box {
        border: 2px dashed #94a3b8;
        border-radius: 12px;
        padding: 1.5rem;
        background-color: #f1f5f9;
        text-align: center;
        color: #475569;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .upload-box:hover {
        background-color: #e2e8f0;
        border-color: #3b82f6;
        color: #1e40af;
    }

    .result-container {
        text-align: center;
        margin-top: 2rem;
    }

    .caption {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 500;
        margin-top: 0.8rem;
    }

    hr {
        border: 0;
        border-top: 2px solid #e2e8f0;
        margin: 2rem 0;
    }

    .resized-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        width: 70%;
        max-width: 500px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧩 Editor de Imágenes | Rotación y Escala")
st.markdown("<p class='subtitle'>Rota y escala tus imágenes de forma interactiva 🎛️</p>", unsafe_allow_html=True)

st.markdown("<div class='upload-box'>📤 Sube una imagen para comenzar</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "png", "jpeg", "webp"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        angle = st.slider("Ángulo de rotación (°)", -180, 180, 30)
    with col2:
        scale = st.slider("Factor de escala", 0.1, 2.0, 1.0, 0.1)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🖼️ Resultado")

    # Convertir la imagen a formato RGB y mostrarla con tamaño controlado
    rgb_img = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    st.markdown(
        f"""
        <div class='result-container'>
            <img src='data:image/png;base64,{st.image(rgb_img, output_format="PNG", use_container_width=False)}' 
                 class='resized-img'>
            <p class='caption'>🔁 Rotada {angle}° y escalada {scale:.1f}x</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("👆 Sube una imagen para editarla.")

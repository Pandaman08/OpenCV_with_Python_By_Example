import cv2
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image

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

def convert_image_to_bytes(img, format='PNG'):

    if len(img.shape) == 2:
        img_pil = Image.fromarray(img)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    
    buf = BytesIO()
    img_pil.save(buf, format=format)
    buf.seek(0)
    return buf

uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    equ = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(img_rgb, caption="Original", use_container_width=True)
        st.download_button(
            label="⬇️ Descargar Original",
            data=convert_image_to_bytes(img),
            file_name="original.png",
            mime="image/png"
        )
    with col2:
        st.image(equ, caption="Contraste Mejorado", use_container_width=True)
        st.download_button(
            label="⬇️ Descargar Contraste",
            data=convert_image_to_bytes(equ),
            file_name="contraste_mejorado.png",
            mime="image/png"
        )
    
    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.image(cl1, caption="CLAHE Aplicado", use_container_width=True)
        st.download_button(
            label="⬇️ Descargar CLAHE",
            data=convert_image_to_bytes(cl1),
            file_name="clahe.png",
            mime="image/png"
        )
    with col4:
        st.image(binary, caption="Binarizada", use_container_width=True)
        st.download_button(
            label="⬇️ Descargar Binarizada",
            data=convert_image_to_bytes(binary),
            file_name="binarizada.png",
            mime="image/png"
        )
    
    st.markdown("---")
    st.markdown("### 📦 Descargar todas las imágenes")
    
    import zipfile
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("original.png", convert_image_to_bytes(img).getvalue())
        zip_file.writestr("contraste_mejorado.png", convert_image_to_bytes(equ).getvalue())
        zip_file.writestr("clahe.png", convert_image_to_bytes(cl1).getvalue())
        zip_file.writestr("binarizada.png", convert_image_to_bytes(binary).getvalue())
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="📥 Descargar todas las imágenes (ZIP)",
        data=zip_buffer,
        file_name="imagenes_procesadas.zip",
        mime="application/zip"
    )
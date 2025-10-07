import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="🎨 Cartoonizador de Imágenes",
    layout="centered"
)

st.title("🎨 Cartoonizador de Imágenes")
st.markdown("""
Sube una imagen y conviértela en:
- 🖌️ **Boceto a lápiz**
- 🎨 **Caricatura a color**
""")


def cartoonize_image(img, sketch_mode=False):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    img_small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    for _ in range(5):
        img_small = cv2.bilateralFilter(img_small, 9, 9, 7)
    img_cartoon = cv2.resize(img_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.bitwise_and(img_cartoon, img_cartoon, mask=mask)

uploaded_file = st.file_uploader("📤 Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.image(pil_image, caption="📸 Imagen original", use_container_width=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2: 
        sketch_btn = st.button("🖌️ Modo Boceto", use_container_width=True)
        cartoon_btn = st.button("🎨 Modo Caricatura", use_container_width=True)

    if sketch_btn or cartoon_btn:
        with st.spinner("🎨 Procesando..."):
            result = cartoonize_image(img, sketch_mode=sketch_btn)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)

            st.image(result_pil, caption=f"🖼️ Resultado: {'Boceto' if sketch_btn else 'Caricatura'}", use_container_width=True)
            st.download_button(
                label="📥 Descargar imagen",
                data=cv2.imencode('.png', cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))[1].tobytes(),
                file_name=f"{'boceto' if sketch_btn else 'caricatura'}.png",
                mime="image/png"
            )
else:
    st.info("👆 Sube una imagen para comenzar.")

st.markdown("""
---
📚 **Basado en**: *OpenCV 3.x with Python By Example* – Capítulo 3: Cartoonizing an Image  
👨‍💻 **Tarea de Sistemas Inteligentes** – Universidad Nacional de Trujillo
""")
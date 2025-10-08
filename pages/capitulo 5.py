import cv2
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="🧠 Detección de esquinas", layout="centered")
st.title("🔍 Detección de Esquinas con Harris")

st.markdown("""
Sube una imagen y observa cómo se detectan sus **esquinas** utilizando el algoritmo de Harris.
""")

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png"])

if uploaded:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = img.copy()

    # Convertir a gris y aplicar Harris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    # Umbral para detección de esquinas
    threshold = 0.01 * dst.max()
    corner_points = np.argwhere(dst > threshold)

    # Dibujar círculos y registrar coordenadas
    coords = []
    for y, x in corner_points:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        coords.append({"X": int(x), "Y": int(y)})

    # Mostrar imágenes lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                 caption="Imagen original",
                 use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"Esquinas detectadas: {len(coords)}",
                 use_container_width=True)

    # Limitar tamaño visual máximo de imágenes
    st.markdown("""
        <style>
        img {
            max-width: 400px !important;
            height: auto;
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Mostrar tabla con coordenadas (máximo 50 para no saturar)
    if len(coords) > 0:
        df = pd.DataFrame(coords)
        st.subheader("📍 Coordenadas de esquinas detectadas")
        st.dataframe(df.head(50), use_container_width=True)
        if len(coords) > 50:
            st.info(f"Mostrando las primeras 50 de {len(coords)} esquinas detectadas.")
    else:
        st.warning("No se detectaron esquinas en esta imagen.")

    # Botón de descarga
    st.download_button(
        label="📥 Descargar imagen con esquinas detectadas",
        data=cv2.imencode('.png', img)[1].tobytes(),
        file_name="esquinas_detectadas.png",
        mime="image/png"
    )

else:
    st.info("👆 Sube una imagen para comenzar.")

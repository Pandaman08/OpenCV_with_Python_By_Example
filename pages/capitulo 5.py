import cv2
import streamlit as st
import numpy as np

uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png"])
if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Puntos rojos
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Esquinas detectadas", use_container_width=True)
    st.download_button(
        label="📥 Descargar imagen con esquinas",
        data=cv2.imencode('.png', img)[1].tobytes(),
        file_name="esquinas_detectadas.png",
        mime="image/png"
    )
else:
    st.info("👆 Sube una imagen para comenzar.")
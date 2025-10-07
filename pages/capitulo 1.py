import cv2
import streamlit as st
import numpy as np

uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png"])
if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    angle = st.slider("Ángulo de rotación", -180, 180, 30)
    scale = st.slider("Factor de escala", 0.1, 2.0, 1.0, 0.1)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))

    st.image(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), caption=f"Rotado {angle}° y escalado {scale}x", use_container_width=True)
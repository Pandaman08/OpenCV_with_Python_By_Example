import cv2
import streamlit as st
import numpy as np

uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png"])
if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    st.image(equ, caption="Contraste mejorado", use_container_width=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    st.image(cl1, caption="CLAHE aplicado", use_container_width=True)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    st.image(binary, caption="Imagen binarizada", use_container_width=True)

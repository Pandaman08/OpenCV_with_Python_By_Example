import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title=" rostro", layout="centered")
st.title(" rostro - Detección de Rostros")

st.markdown("""
Sube una imagen y detecta rostros usando Haar Cascades.
""")

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"✅ {len(faces)} rostros detectados", use_container_width=True)

    st.download_button(
        label="📥 Descargar imagen con rostros",
        data=cv2.imencode('.png', img)[1].tobytes(),
        file_name="rostros_detectados.png",
        mime="image/png"
    )
else:
    st.info("👆 Sube una imagen para comenzar.")
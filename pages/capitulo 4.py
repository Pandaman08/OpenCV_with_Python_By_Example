import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="🧠 Detección de Rostros", layout="centered")

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 30px;
            color: #4A90E2;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 16px;
            margin-bottom: 25px;
        }
        .stImage img {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 75%;
            height: auto;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-top: 35px;
        }
        .footer hr {
            border: none;
            border-top: 1px solid #ccc;
            margin: 20px auto;
            width: 80%;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Detección de Rostros</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sube una imagen y detecta rostros usando Haar Cascades.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption=f"✅ {len(faces)} rostro(s) detectado(s)",
        use_container_width=True
    )

    st.download_button(
        label="📥 Descargar imagen con rostros",
        data=cv2.imencode('.png', img)[1].tobytes(),
        file_name="rostros_detectados.png",
        mime="image/png"
    )
else:
    st.info("👆 Sube una imagen para comenzar.")

st.markdown("""
<div class="footer">
    <hr>
    📚 <b>Tecnología:</b> OpenCV + Streamlit<br>
    👨‍💻 <b>Proyecto de Sistemas Inteligentes</b> — Universidad Nacional de Trujillo
</div>
""", unsafe_allow_html=True)

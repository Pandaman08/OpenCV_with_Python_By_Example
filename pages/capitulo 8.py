import streamlit as st
import cv2
import numpy as np

st.title("🎨 Rastreo por Color + Contornos")

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    color = st.selectbox("Elige un color", ["Rojo", "Azul", "Verde", "Amarillo"])

    ranges = {
        "Rojo": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        "Azul": (np.array([100, 100, 100]), np.array([130, 255, 255])),
        "Verde": (np.array([40, 100, 100]), np.array([80, 255, 255])),
        "Amarillo": (np.array([20, 100, 100]), np.array([30, 255, 255]))
    }
    
    lower, upper = ranges[color]
    mask = cv2.inRange(hsv, lower, upper)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar ruido
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Objetos {color} detectados", use_container_width=True)
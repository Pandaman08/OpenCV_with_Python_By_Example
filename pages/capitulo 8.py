import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Rastreo de Color", page_icon="🎨", layout="wide")
st.title("🎨 Rastreo por Color + Contornos")

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png"])

if uploaded:
    # --- Leer imagen ---
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > 800:  # Escalar imágenes grandes para mejorar rendimiento
        scale = 800 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Selección de color ---
    color = st.selectbox("🎯 Elige un color a rastrear", ["Rojo", "Azul", "Verde", "Amarillo"])

    # Rango de colores en HSV
    ranges = {
        "Rojo": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                 (np.array([160, 100, 100]), np.array([179, 255, 255]))],  # Dos rangos para rojo
        "Azul": [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
        "Verde": [(np.array([40, 70, 70]), np.array([80, 255, 255]))],
        "Amarillo": [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
    }

    # --- Crear máscara combinada (por si hay varios rangos, como el rojo) ---
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in ranges[color]:
        mask |= cv2.inRange(hsv, lower, upper)

    # --- Limpiar ruido con operaciones morfológicas ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Encontrar contornos ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()

    detected = 0
    for contour in contours:
        if cv2.contourArea(contour) > 400:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, color, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detected += 1

    # --- Mostrar resultados ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="📸 Imagen original",
                 use_container_width=True)
    with col2:
        st.image(mask, caption=f"🎭 Máscara de color ({color})", use_container_width=True)
    with col3:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption=f"🟩 Objetos {color.lower()} detectados ({detected})",
                 use_container_width=True)

    st.success(f"✅ Se detectaron {detected} objeto(s) de color {color.lower()}.")
else:
    st.info("👆 Sube una imagen para comenzar.")

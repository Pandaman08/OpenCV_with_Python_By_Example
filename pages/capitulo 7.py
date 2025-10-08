import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Segmentación GrabCut", page_icon="🎯", layout="wide")
st.title("🎯 Segmentación con GrabCut")

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png"])

if uploaded:
    # --- Leer y redimensionar si es muy grande ---
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    max_dim = 800
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    # --- Sliders para el rectángulo ---
    st.info("💡 Ajusta el rectángulo para seleccionar el objeto a segmentar")

    x_pct = st.slider("X inicial (%)", 0, 100, 20) / 100
    y_pct = st.slider("Y inicial (%)", 0, 100, 20) / 100
    w_pct = st.slider("Ancho (%)", 10, 100, 40) / 100
    h_pct = st.slider("Alto (%)", 10, 100, 40) / 100

    x = int(x_pct * w)
    y = int(y_pct * h)
    w_rect = int(w_pct * w)
    h_rect = int(h_pct * h)

    # Asegurar que el rectángulo no salga de los límites
    w_rect = min(w_rect, w - x - 1)
    h_rect = min(h_rect, h - y - 1)

    # --- Aplicar GrabCut ---
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (x, y, w_rect, h_rect)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = img * mask2[:, :, np.newaxis]
    except cv2.error as e:
        st.error(f"❌ Error al ejecutar GrabCut: {e}")
        st.stop()

    # --- Mostrar resultados ---
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB),
                 caption="🟩 ROI seleccionado",
                 use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption="✂️ Segmentación resultante",
                 use_container_width=True)

    # --- Botón de descarga ---
    st.download_button(
        label="📥 Descargar resultado segmentado",
        data=cv2.imencode('.png', result)[1].tobytes(),
        file_name="segmentacion_grabcut.png",
        mime="image/png"
    )

else:
    st.info("👆 Sube una imagen para comenzar.")

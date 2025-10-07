import streamlit as st
import cv2
import numpy as np

st.title("🎯 Segmentación con GrabCut")

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    
    st.info("💡 Ajusta el rectángulo para seleccionar el objeto a segmentar")

    x_pct = st.slider("X inicial (%)", 0, 100, 20) / 100
    y_pct = st.slider("Y inicial (%)", 0, 100, 20) / 100
    w_pct = st.slider("Ancho (%)", 10, 100, 40) / 100
    h_pct = st.slider("Alto (%)", 10, 100, 40) / 100
    
    x = int(x_pct * w)
    y = int(y_pct * h)
    w_rect = int(w_pct * w)
    h_rect = int(h_pct * h)

    w_rect = min(w_rect, w - x)
    h_rect = min(h_rect, h - y)
    
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (x, y, w_rect, h_rect)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = img * mask2[:, :, np.newaxis]
    
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x, y), (x+w_rect, y+h_rect), (0, 255, 0), 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB), caption="ROI seleccionado", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Segmentación", use_container_width=True)
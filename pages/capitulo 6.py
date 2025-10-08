import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="🧵 Seam Carving Optimizado", layout="centered")

st.title("🧵 Reducción Inteligente de Ancho (Optimizada)")
st.markdown("""
Esta versión usa **operaciones vectorizadas con NumPy** para acelerar el cálculo de costuras.
""")

uploaded = st.file_uploader("📤 Sube una imagen (JPG/PNG)", type=["jpg", "png"])

def compute_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(np.abs(grad_x) + np.abs(grad_y))

def find_seam(energy):
    rows, cols = energy.shape
    M = energy.astype(np.float64)
    backtrack = np.zeros_like(M, dtype=np.int32)

    for i in range(1, rows):
        left = np.roll(M[i-1], 1)
        right = np.roll(M[i-1], -1)
        stacked = np.vstack([left, M[i-1], right])
        idx = np.argmin(stacked, axis=0)
        M[i] += stacked[idx, np.arange(cols)]
        backtrack[i] = idx - 1
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    for i in range(rows - 2, -1, -1):
        seam[i] = np.clip(seam[i+1] + backtrack[i+1, seam[i+1]], 0, energy.shape[1]-1)
    return seam

def remove_seam(img, seam):
    rows, cols = img.shape[:2]
    mask = np.ones((rows, cols), dtype=bool)
    mask[np.arange(rows), seam] = False
    return img[mask].reshape((rows, cols - 1, 3))

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        num_seams = st.slider("Número de costuras a eliminar", 1, 100, 40)
    with col2:
        scale_factor = st.selectbox("Reducir resolución", ["100%", "75%", "50%"], index=1)

    scale = {"100%": 1.0, "75%": 0.75, "50%": 0.5}[scale_factor]
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    original = img.copy()
    img_seams = img.copy()
    result = img.copy()

    progress = st.progress(0, text="Procesando...")

    for i in range(num_seams):
        energy = compute_energy(result)
        seam = find_seam(energy)

        img_seams[np.arange(len(seam)), seam] = [0, 255, 0]
        result = remove_seam(result, seam)

        progress.progress((i + 1) / num_seams, text=f"Costura {i + 1}/{num_seams}")

    progress.empty()
    st.success(f"✅ {num_seams} costuras eliminadas en una imagen escalada al {scale_factor}")

    colA, colB = st.columns(2)
    with colA:
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="📸 Original", use_container_width=True)
    with colB:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"🧵 Reducción ({num_seams} costuras)", use_container_width=True)

    st.image(cv2.cvtColor(img_seams, cv2.COLOR_BGR2RGB), caption="🟢 Costuras detectadas", use_container_width=True)

    st.download_button(
        label="📥 Descargar imagen reducida",
        data=cv2.imencode('.png', result)[1].tobytes(),
        file_name="imagen_reducida.png",
        mime="image/png"
    )
else:
    st.info("👆 Sube una imagen para comenzar.")

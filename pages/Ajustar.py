import streamlit as st
import cv2
import numpy as np
from PIL import Image

def resize_with_content_aware_padding(
    image: np.ndarray,
    target_width_px: int,
    target_height_px: int,
    border_mode=cv2.BORDER_REPLICATE
) -> np.ndarray:
    """
    Rellena la imagen hasta las dimensiones objetivo usando un modo de borde inteligente.
    - BORDER_REPLICATE: repite los bordes (bueno para pasto/cielo).
    - BORDER_REFLECT_101: refleja el contenido (evita bordes duros).
    """
    h, w = image.shape[:2]

    if target_width_px <= w or target_height_px <= h:
        # Si el objetivo es más pequeño, recortamos (o podrías escalar)
        # Pero en tu caso, asumimos que siempre es más grande
        st.warning("Las dimensiones objetivo son más pequeñas que la imagen original. Se recortará.")
        x1 = max(0, (w - target_width_px) // 2)
        y1 = max(0, (h - target_height_px) // 2)
        return image[y1:y1 + target_height_px, x1:x1 + target_width_px]

    # Calcular cuánto rellenar en cada lado
    delta_w = target_width_px - w
    delta_h = target_height_px - h

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Aplicar relleno "inteligente"
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, border_mode)
    return padded

def mm_to_pixels(mm: float, dpi: int = 300) -> int:
    return int(mm / 25.4 * dpi)

def cm_to_pixels(cm: float, dpi: int = 300) -> int:
    return int(cm / 2.54 * dpi)

# Interfaz de Streamlit
st.title("Convertir Imagen a Marco Vertical con Relleno Inteligente")
st.write("""
Sube una imagen y define las dimensiones del marco vertical en **cm o mm**.
La imagen se centrará y se rellenará usando los bordes existentes (pasto, cielo, etc.).
""")

uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer imagen
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Mostrar imagen original
    st.subheader("Imagen original")
    st.image(pil_image, caption="Imagen subida por el usuario", use_container_width=True)
    # Parámetros de salida
    st.sidebar.header("Configuración de salida")
    unit = st.sidebar.radio("Unidad de medida", ("cm", "mm"), index=0)
    dpi = st.sidebar.number_input("DPI (resolución)", min_value=72, max_value=600, value=300, step=50)

    if unit == "cm":
        width_input = st.sidebar.number_input("Ancho (cm)", min_value=1.0, value=21.0, step=1.0)  # A4 width
        height_input = st.sidebar.number_input("Alto (cm)", min_value=1.0, value=29.7, step=1.0)  # A4 height
        w_px = cm_to_pixels(width_input, dpi)
        h_px = cm_to_pixels(height_input, dpi)
    else:
        width_input = st.sidebar.number_input("Ancho (mm)", min_value=10.0, value=210.0, step=10.0)
        height_input = st.sidebar.number_input("Alto (mm)", min_value=10.0, value=297.0, step=10.0)
        w_px = mm_to_pixels(width_input, dpi)
        h_px = mm_to_pixels(height_input, dpi)

    border_mode = st.sidebar.selectbox(
        "Método de relleno",
        options=[
            ("Repetir bordes (BORDER_REPLICATE)", cv2.BORDER_REPLICATE),
            ("Reflejar (BORDER_REFLECT_101)", cv2.BORDER_REFLECT_101),
            ("Reflejar simple (BORDER_REFLECT)", cv2.BORDER_REFLECT),
        ],
        format_func=lambda x: x[0]
    )[1]

    # Procesar
    result = resize_with_content_aware_padding(image, w_px, h_px, border_mode=border_mode)

    # Mostrar (reducir tamaño para vista previa)
    max_display = 800
    h_disp, w_disp = result.shape[:2]
    scale = min(max_display / max(h_disp, w_disp), 1.0)
    preview = cv2.resize(result, (int(w_disp * scale), int(h_disp * scale)))

    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Vista previa (no a escala real)", use_container_width=True)

    # Descargar
    is_success, buffer = cv2.imencode(".png", result)
    if is_success:
        st.download_button(
            label="📥 Descargar imagen en tamaño real",
            data=buffer.tobytes(),
            file_name="imagen_vertical_extendida.png",
            mime="image/png"
        )

    st.info(f"Dimensiones finales: {w_px} × {h_px} píxeles ({width_input} × {height_input} {unit}, DPI: {dpi})")

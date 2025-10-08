import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Emparejamiento ORB", page_icon="🔍", layout="centered")
st.title("🔍 ORB - Emparejamiento de Características")

st.markdown("""
Este ejercicio implementa el **emparejamiento de características con ORB**  
(siguiendo el *Capítulo 9: Object Recognition* de *OpenCV 3.x with Python By Example*).

Sube dos imágenes:
- 🖼️ **Imagen 1:** objeto a buscar (por ejemplo, un logo o libro)  
- 🌄 **Imagen 2:** escena donde podría encontrarse ese objeto
""")

col1, col2 = st.columns(2)
with col1:
    img1_file = st.file_uploader("📤 Imagen 1 (Objeto)", type=["jpg", "png"], key="img1")
with col2:
    img2_file = st.file_uploader("📤 Imagen 2 (Escena)", type=["jpg", "png"], key="img2")

if img1_file and img2_file:
    img1 = cv2.imdecode(np.frombuffer(img1_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        st.error("❌ Error al cargar las imágenes.")
    else:
        # Redimensionar imágenes grandes
        max_dim = 600
        for i, img in enumerate([img1, img2]):
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                if i == 0:
                    img1 = resized
                else:
                    img2 = resized

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            st.warning("⚠️ No se encontraron características suficientes en una de las imágenes.")
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Dibujar puntos clave individuales
            img_kp1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
            img_kp2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

            # Mostrar imágenes con puntos clave detectados
            st.markdown("### 🔹 Puntos clave detectados")
            col3, col4 = st.columns(2)
            with col3:
                st.image(cv2.cvtColor(img_kp1, cv2.COLOR_BGR2RGB), caption=f"{len(kp1)} puntos en Imagen 1", use_container_width=True)
            with col4:
                st.image(cv2.cvtColor(img_kp2, cv2.COLOR_BGR2RGB), caption=f"{len(kp2)} puntos en Imagen 2", use_container_width=True)

            # Dibujar coincidencias
            img_matches = cv2.drawMatches(
                img1, kp1, img2, kp2, matches[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            st.markdown("### 🔸 Coincidencias ORB")
            st.image(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB),
                     caption=f"✅ {len(matches)} coincidencias encontradas",
                     use_container_width=True)

            # Evaluación
            if len(matches) > 25:
                st.success("🎯 ¡Objeto detectado en la escena con buena confianza!")
            elif 10 < len(matches) <= 25:
                st.warning("⚠️ Coincidencias moderadas. Puede que el objeto esté parcialmente visible.")
            else:
                st.error("❌ No se encontró el objeto (pocas coincidencias).")
else:
    st.info("👆 Sube dos imágenes para comenzar.")

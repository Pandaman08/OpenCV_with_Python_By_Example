# ejercicio_9_orb.py
import streamlit as st
import cv2
import numpy as np

st.title("🔍 ORB - Emparejamiento de Características")

st.markdown("""
Este ejercicio implementa el emparejamiento de características con ORB,  
siguiendo los conceptos del **Capítulo 9: Object Recognition** del libro  
*OpenCV 3.x with Python By Example*.

Sube dos imágenes:
- **Imagen 1**: el objeto que quieres buscar (ej: logo, libro).
- **Imagen 2**: una escena donde podría estar ese objeto.
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
            
            img_matches = cv2.drawMatches(
                img1, kp1, img2, kp2, matches[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            st.image(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB), 
                     caption=f"✅ {len(matches)} coincidencias encontradas", 
                     use_container_width=True)
            
            if len(matches) > 10:
                st.success("🎯 ¡Objeto detectado en la escena!")
            else:
                st.warning("⚠️ No se encontró el objeto (pocas coincidencias).")
else:
    st.info("👆 Sube dos imágenes para comenzar.")
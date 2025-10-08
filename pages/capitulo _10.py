import streamlit as st
import cv2
import numpy as np

st.title("📦 Detección de Marcadores ArUco")

uploaded = st.file_uploader("📤 Sube una imagen con marcador ArUco", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        try:
            parameters = cv2.aruco.DetectorParameters()
        except AttributeError:
            parameters = cv2.aruco.DetectorParameters_create()

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            img_aruco = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
            st.image(cv2.cvtColor(img_aruco, cv2.COLOR_BGR2RGB),
                     caption=f"✅ {len(ids)} marcadores detectados: {ids.flatten().tolist()}",
                     use_container_width=True)
        else:
            st.warning("⚠️ No se detectaron marcadores ArUco.")
    else:
        st.error("❌ No se pudo procesar la imagen. Intenta con otro formato.")

import streamlit as st
import cv2
import numpy as np

st.title("📦 Detección de Marcador ArUco")

uploaded = st.file_uploader("📤 Sube una imagen con marcador ArUco", type=["jpg", "png"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        img_aruco = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
        st.image(cv2.cvtColor(img_aruco, cv2.COLOR_BGR2RGB), 
                 caption=f"✅ {len(ids)} marcadores detectados", 
                 use_container_width=True)
    else:
        st.warning("⚠️ No se detectaron marcadores ArUco")
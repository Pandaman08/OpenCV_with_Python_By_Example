import streamlit as st
import cv2
import numpy as np

st.title("🧵 Seam Carving - Reducción de Ancho")
st.write("El proceso tarda más tiempo dependiendo del número de costuras a eliminar.")

uploaded = st.file_uploader("📤 Sube una imagen", type=["jpg", "png"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    
    num_seams = st.slider("Número de costuras a eliminar", 1, 100, 20)
    
    def compute_energy_matrix(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, 
                               cv2.convertScaleAbs(sobel_y), 0.5, 0)
    
    def find_vertical_seam(img, energy):
        rows, cols = img.shape[:2]
        seam = np.zeros(rows, dtype=int)
        dist_to = np.zeros(img.shape[:2]) + float('inf')
        dist_to[0, :] = np.zeros(cols)
        edge_to = np.zeros(img.shape[:2])
        
        for row in range(rows - 1):
            for col in range(cols):
                if col != 0 and dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]:
                    dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1]
                    edge_to[row+1, col-1] = 1
                if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]:
                    dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col]
                    edge_to[row+1, col] = 0
                if col != cols-1 and dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]:
                    dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1]
                    edge_to[row+1, col+1] = -1
        
        seam[rows-1] = np.argmin(dist_to[rows-1, :])
        for i in range(rows-1, 0, -1):
            seam[i-1] = seam[i] + edge_to[i, int(seam[i])]
        
        return seam
    
    def remove_vertical_seam(img, seam):
        rows, cols = img.shape[:2]
        for row in range(rows):
            for col in range(int(seam[row]), cols-1):
                img[row, col] = img[row, col+1]
        return img[:, 0:cols-1]
    
    result = img.copy()
    img_with_seams = img.copy()
    
    for i in range(num_seams):
        energy = compute_energy_matrix(result)
        seam = find_vertical_seam(result, energy)
        
        for row in range(img_with_seams.shape[0]):
            if 0 <= int(seam[row]) < img_with_seams.shape[1]:
                img_with_seams[row, int(seam[row])] = [0, 255, 0]
        
        result = remove_vertical_seam(result, seam)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"{num_seams} costuras eliminadas", use_container_width=True)
    
    st.image(cv2.cvtColor(img_with_seams, cv2.COLOR_BGR2RGB), caption="Costuras detectadas", use_container_width=True)
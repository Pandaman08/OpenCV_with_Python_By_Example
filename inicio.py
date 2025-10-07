# app_inicio.py
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="👋 Bienvenido a OpenCV + Streamlit",
    page_icon="🖼️",
    layout="centered"
)

# Título principal
st.title("👋 Bienvenido a *OpenCV 3.x con Python By Example*")
st.markdown("### 🧪 Aplicaciones interactivas basadas en el libro")

# Descripción general
st.markdown("""
Este proyecto presenta **11 aplicaciones interactivas** desarrolladas con **OpenCV y Streamlit**, 
cada una inspirada en un capítulo del libro:

> **OpenCV 3.x with Python By Example**  
> *Gabriel Garrido, Prateek Joshi*

Todas las aplicaciones funcionan **sin cámara en tiempo real** (por limitaciones de Streamlit Cloud)  
y se basan en **procesamiento de imágenes estáticas subidas por el usuario**.
""")

# Lista de ejercicios
st.subheader("📚 Ejercicios implementados")

ejercicios = {
    "1. Transformaciones Geométricas": "Capítulo 1 – Rotación, escalado y traslación.",
    "2. Mejora de Contraste": "Capítulo 2 – Ecualización de histograma en YUV.",
    "3. Cartoonizador de Imágenes": "Capítulo 3 – Convierte fotos en caricaturas o bocetos.",
    "4. Detección de rostro": "Capitulo 4 - Detecta rostros en imágenes estáticas (Haar Cascade)",
    "5. Esquinas con Harris": "Capítulo 5 – Detección de esquinas en imágenes.",
    "6. Seam Carving": "Capítulo 6 – Reducción de tamaño de imagen sin distorsión.",
    "7. Segmentación con GrabCut": "Capítulo 7 – Selección interactiva y segmentación.",
    "8. Rastreo por color": "Capítulo 8 – Rastreo por color en imágenes estáticas.",
    "9. Emparejamiento de características con ORB": "Capítulo 9 – Rotación, escalado y traslación." ,
    "10. Detección de Marcador ArUco": "Capítulo 10 – Realidad Aumentada .",
    "11. Clasificador con ANN (Red Neuronal)": "Capítulo 11 – HOG + Red Neuronal MLP para clasificar perros vs gatos."
}

for nombre, desc in ejercicios.items():
    st.markdown(f"- **{nombre}**: {desc}")

st.info("""
💡 **Nota**: Estas aplicaciones están diseñadas para ejecutarse en **Streamlit Cloud**,  
por lo que **no usan la cámara en tiempo real**, sino imágenes subidas manualmente.
""")

st.markdown("""
---
👨‍💻 **Desarrollado por**: Alvaro Paz Romero 
🏛️ **Curso**: Sistemas Inteligentes – Universidad Nacional de Trujillo  
📘 **Base teórica**: *OpenCV 3.x with Python By Example* (2da edición, 2018)
""")

st.markdown("[Ver código fuente en GitHub🤖](https://github.com/pandaman08/OpenCV_with_Python_By_Example)")
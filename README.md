# 🖼️ OpenCV 3.x con Python – Proyectos Interactivos

Este repositorio contiene **11 aplicaciones interactivas** desarrolladas con **OpenCV y Streamlit**, cada una basada en un capítulo del libro:

> **OpenCV 3.x with Python By Example**  
> *Gabriel Garrido, Prateek Joshi (2ª edición, 2018)*

Todas las aplicaciones están diseñadas para funcionar en **Streamlit Cloud**, por lo que **no requieren acceso a la cámara en tiempo real**. En su lugar, procesan **imágenes subidas por el usuario**, lo que permite una demostración clara, reproducible y compatible con la nube.

---

## 📚 Capítulos implementados

| N° | Aplicación | Capítulo | Descripción |
|----|------------|----------|-------------|
| 1 | 🎨 **Cartoonizador de Imágenes** | Capítulo 3 | Convierte fotos en caricaturas o bocetos a lápiz usando filtros de mediana, Laplacian y bilateral. |
| 2 | 🔍 **Detección de Bordes** | Capítulo 2 | Aplica Sobel y Laplacian para detectar bordes en imágenes. |
| 3 | 🌀 **Filtro Gaussiano** | Capítulo 2 | Suaviza imágenes con control del tamaño del kernel. |
| 4 | 🎨 **Efecto Acuarela** | Capítulo 3 | Estiliza imágenes con `cv2.stylization`. |
| 5 | ✨ **Filtro Viñeta** | Capítulo 2 | Crea un efecto de enfoque central con degradado suave. |
| 6 | 📈 **Mejora de Contraste** | Capítulo 2 | Ecualiza el histograma en espacio YUV para mejorar el contraste. |
| 7 | 📍 **Esquinas con Harris** | Capítulo 5 | Detecta esquinas usando el detector de Harris. |
| 8 | ✂️ **Segmentación con GrabCut** | Capítulo 7 | Segmenta objetos mediante selección de región y algoritmo GrabCut. |
| 9 | 🔄 **Transformaciones Geométricas** | Capítulo 1 | Aplica rotación, escalado y traslación a imágenes. |
| 10 | 📐 **Detección de Contornos** | Capítulo 7 | Encuentra y dibuja contornos de formas en imágenes. |
| 11 | 🧠 **Clasificador con Red Neuronal (ANN)** | Capítulo 11 | Usa HOG + ANN (MLP) para clasificar perros vs gatos. |
.

---

## 🚀 Cómo ejecutar localmente

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Pandaman08/OpenCV_with_Python_By_Example.git
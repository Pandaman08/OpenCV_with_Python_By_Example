# 🖼️ OpenCV 3.x con Python – Proyectos Interactivos

Este repositorio contiene **11 aplicaciones interactivas** desarrolladas con **OpenCV y Streamlit**, cada una basada en un capítulo del libro:

> **OpenCV 3.x with Python By Example**  
> *Gabriel Garrido, Prateek Joshi (2ª edición, 2018)*

Todas las aplicaciones están diseñadas para funcionar en **Streamlit Cloud**, por lo que **no requieren acceso a la cámara en tiempo real**. En su lugar, procesan **imágenes subidas por el usuario**, lo que permite una demostración clara, reproducible y compatible con la nube.

---

## 📚 Capítulos implementados

| N° | Aplicación | Capítulo | Descripción |
|----|------------|----------|-------------|
| 1 | 🔄 **Transformaciones Geométricas** | Capítulo 1 | Rotación, escalado y traslación de imágenes. |
| 2 | 📈 **Mejora de Contraste** | Capítulo 2 | Ecualización de histograma en espacio de color YUV. |
| 3 | 🎨 **Cartoonizador de Imágenes** | Capítulo 3 | Convierte fotos en caricaturas o bocetos a lápiz. |
| 4 | 👤 **Detección de rostro** | Capítulo 4 | Detecta rostros en imágenes estáticas usando Haar Cascade. |
| 5 | 📍 **Esquinas con Harris** | Capítulo 5 | Detección de esquinas en imágenes mediante el detector de Harris. |
| 6 | ✂️ **Seam Carving** | Capítulo 6 | Reducción de tamaño de imagen sin distorsionar regiones importantes. |
| 7 | 🖌️ **Segmentación con GrabCut** | Capítulo 7 | Segmentación interactiva de objetos mediante selección de región. |
| 8 | 🎯 **Rastreo por color** | Capítulo 8 | Detección y visualización de objetos por rango de color (HSV). |
| 9 | 🔗 **Emparejamiento de características con ORB** | Capítulo 9 | Emparejamiento de puntos clave entre dos imágenes usando ORB. |
| 10 | 🕶️ **Detección de Marcador ArUco** | Capítulo 10 | Realidad Aumentada mediante detección de marcadores ArUco. |
| 11 | 🧠 **Clasificador con ANN (Red Neuronal)** | Capítulo 11 | Clasificación de perros vs gatos usando HOG + Red Neuronal MLP. |

> ✅ **Nota sobre el Capítulo 10**:  
> Aunque el libro no menciona explícitamente **ArUco**, el Capítulo 10 trata sobre **Realidad Aumentada**, estimación de pose (`solvePnP`) y superposición de objetos virtuales.  
> **ArUco es una implementación moderna, robusta y estándar** que aplica exactamente esos principios, por lo que se considera una extensión válida y práctica del contenido del capítulo.

---

## 🚀 Cómo ejecutar localmente

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/openCV-python-proyectos.git
   cd openCV-python-proyectos
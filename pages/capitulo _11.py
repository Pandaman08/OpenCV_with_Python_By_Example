import streamlit as st
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧠 Clasificador de Perros y Gatos (HOG + ANN)")

st.markdown("""
Se usa **HOG + Red Neuronal Artificial (ANN)** con OpenCV.

🔧 **Características**:
- Red neuronal de 3 capas (entrada, oculta, salida)
- Escalado de características para mejor convergencia
- Aumento de datos (rotación y espejado)
- Entrenamiento con backpropagation
- Cálculo de confianza mediante softmax

⚠️ **Limitaciones**:
- Funciona mejor con imágenes simples y en vista frontal.  
- Requiere suficientes imágenes de entrenamiento.
""")


class_a_files = [
    "imagenes/perro01.jpg",
    "imagenes/perro02.jpg",
    "imagenes/perro03.jpg",
    "imagenes/perro04.jpg",
    "imagenes/perro05.webp",
    "imagenes/perro06.jpg",
    "imagenes/perro07.jpeg",
    "imagenes/perro08.jpg",
    "imagenes/perro09.jpeg",
]

class_b_files = [
    "imagenes/gato01.jpeg",
    "imagenes/gato02.jpg",
    "imagenes/gato03.webp",
    "imagenes/gato04.jpeg",
    "imagenes/gato05.jpg",
    "imagenes/gato06.jpg",
]

def extract_hog(img):
    img = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    return hog.compute(gray).flatten()

def augment_image(img):
    flipped = cv2.flip(img, 1)  
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return [img, flipped, rotated]

# ------------------------
# Cargar y procesar datos
# ------------------------
X, y = [], []

def load_images(file_list, label):
    for f in file_list:
        if os.path.exists(f):
            img = cv2.imread(f)
            if img is not None:
                for variant in augment_image(img):
                    X.append(extract_hog(variant))
                    y.append(label)
        else:
            st.warning(f"⚠️ No se encontró: {f}")

load_images(class_a_files, 0)  
load_images(class_b_files, 1) 

if len(X) >= 6 and len(set(y)) == 2:
    st.info(f"📸 Total de imágenes procesadas: {len(X)}")
    
    # Preparar datos
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    # Codificar etiquetas para ANN (one-hot encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = np.zeros((len(y_encoded), 2), dtype=np.float32)
    y_onehot[np.arange(len(y_encoded)), y_encoded] = 1
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=0.25, random_state=42
    )
    y_test_labels = np.argmax(y_test, axis=1)
    
    # ------------------------
    # Crear y entrenar ANN
    # ------------------------
    ann = cv2.ml.ANN_MLP_create()
    
    # Arquitectura: [input_features, hidden_layer, output_classes]
    input_size = X_train.shape[1]
    hidden_size = 100  # Capa oculta con 100 neuronas
    output_size = 2    # 2 clases: perro y gato
    
    layer_sizes = np.array([input_size, hidden_size, output_size], dtype=np.int32)
    ann.setLayerSizes(layer_sizes)
    
    # Configuración de entrenamiento
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)  # learning rate, momentum
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 1000, 0.01))
    
    # Entrenar
    with st.spinner("🔄 Entrenando red neuronal..."):
        ann.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    
    # ------------------------
    # Evaluar modelo
    # ------------------------
    _, y_pred = ann.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    acc = accuracy_score(y_test_labels, y_pred_labels)
    st.success(f"✅ Modelo entrenado con {acc*100:.1f}% de precisión")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    st.write("📊 Matriz de confusión:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Perro', 'Gato'], 
                yticklabels=['Perro', 'Gato'])
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)
    
    # ------------------------
    # Interfaz de predicción
    # ------------------------
    st.markdown("---")
    st.subheader("🔍 Clasificar nueva imagen")
    
    test_img = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "png", "jpeg", "webp"])
    
    if test_img:
        file_bytes = np.asarray(bytearray(test_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("❌ Imagen no válida.")
        else:
            features = extract_hog(img)
            features_scaled = scaler.transform([features]).astype(np.float32)
            
            _, prediction = ann.predict(features_scaled)
            prediction = prediction[0]

            exp_pred = np.exp(prediction - np.max(prediction))  # Estabilidad numérica
            probas = exp_pred / np.sum(exp_pred)
            
            predicted_class = np.argmax(probas)
            confidence = probas[predicted_class]

            label = "🐶 Perro" if predicted_class == 0 else "🐱 Gato"
            
            if confidence < 0.6:
                caption = f"❓ Predicción incierta: {label} (confianza: {confidence:.2f})"
                st.warning("La red neuronal no está muy segura. Podría no ser ni perro ni gato.")
            else:
                caption = f"🔍 Predicción: {label} (confianza: {confidence:.2f})"

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True
            

else:
    st.error("❌ Necesitas al menos 3 imágenes válidas de **perro** y 3 de **gato**.")
    st.info("Asegúrate de tener las imágenes en la carpeta 'imagenes/' con los nombres correctos.")

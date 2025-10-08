import streamlit as st
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clasificador HOG + ANN", page_icon="🐶", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f9fafb;
            color: #1e293b;
        }
        .main {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        h1, h2, h3, h4 {
            text-align: center;
            color: #334155;
        }
        .metric-box {
            text-align: center;
            border-radius: 12px;
            padding: 1rem;
            background: linear-gradient(135deg, #f1f5f9, #e0f2fe);
            box-shadow: inset 0 0 5px rgba(0,0,0,0.1);
        }
        .metric-label {
            font-size: 1.1rem;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
        }
        hr {
            border: 0;
            border-top: 2px solid #e2e8f0;
            margin: 2rem 0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🐶 Clasificador de Perros y Gatos (HOG + ANN)")

st.markdown("""
<div style='text-align: justify; font-size: 1.05rem;'>
Este modelo combina **Histogram of Oriented Gradients (HOG)** y una **Red Neuronal Artificial (ANN)** 
para clasificar imágenes entre <b>perros 🐶</b> y <b>gatos 🐱</b>.

### 🔍 Características:
- Red neuronal con **3 capas** (entrada, oculta y salida).
- Entrenamiento con **Backpropagation**.
- Escalado de características para mejorar el rendimiento.
- Visualización de la **matriz de confusión**.
</div>
""", unsafe_allow_html=True)


class_a_files = [
    "imagenes/perro01.jpg", "imagenes/perro02.jpg", "imagenes/perro03.jpg",
    "imagenes/perro04.jpg", "imagenes/perro05.webp", "imagenes/perro06.jpg",
    "imagenes/perro07.jpeg", "imagenes/perro08.jpg", "imagenes/perro09.jpeg",
]
class_b_files = [
    "imagenes/gato01.jpeg", "imagenes/gato02.jpg", "imagenes/gato03.webp",
    "imagenes/gato04.jpeg", "imagenes/gato05.jpg", "imagenes/gato06.jpg",
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

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = np.zeros((len(y_encoded), 2), dtype=np.float32)
    y_onehot[np.arange(len(y_encoded)), y_encoded] = 1

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.25, random_state=42)
    y_test_labels = np.argmax(y_test, axis=1)

    ann = cv2.ml.ANN_MLP_create()
    layer_sizes = np.array([X_train.shape[1], 100, 2], dtype=np.int32)
    ann.setLayerSizes(layer_sizes)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 800, 0.01))

    with st.spinner("🧠 Entrenando red neuronal..."):
        ann.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    _, y_pred = ann.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test_labels, y_pred_labels)

    st.success(f"✅ Precisión del modelo: **{acc*100:.2f}%**")

    st.markdown("### 📊 Matriz de Confusión")

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5, 4))  # Tamaño compacto
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        cbar_kws={'shrink': 0.8},
        linewidths=0.5,
        linecolor='white',
        square=True,
        annot_kws={"size": 9, "weight": "bold"},
        xticklabels=['Perro', 'Gato'],
        yticklabels=['Perro', 'Gato'],
        ax=ax
    )

    ax.set_xlabel("Predicción", fontsize=9, labelpad=5)
    ax.set_ylabel("Real", fontsize=9, labelpad=5)
    ax.set_title("Matriz de Confusión", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(axis='both', labelsize=8)

    fig.tight_layout()

    col1, col2, col3 = st.columns([2, 1, 2])  
    with col2:
        st.pyplot(fig, use_container_width=False)

    plt.close(fig)

    st.subheader("🔍 Clasificar imagen")

    test_img = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

    if test_img:
        file_bytes = np.asarray(bytearray(test_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is not None:
            features = extract_hog(img)
            features_scaled = scaler.transform([features]).astype(np.float32)
            _, pred = ann.predict(features_scaled)

            exp_pred = np.exp(pred - np.max(pred))
            probas = exp_pred / np.sum(exp_pred)
            predicted_class = np.argmax(probas)
            confidence = probas[0][predicted_class]

            label = "🐶 Perro" if predicted_class == 0 else "🐱 Gato"

            if confidence < 0.6:
                caption = f"❓ Predicción incierta: {label} ({confidence:.2f})"
                st.warning("⚠️ La red no está muy segura.")
            else:
                caption = f"🔍 Predicción: {label} ({confidence:.2f})"

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption=caption, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='metric-box' style='background:linear-gradient(135deg, #1e3a8a, #2563eb); color:white;'>
                    <div class='metric-label'>🐶 Probabilidad Perro</div>
                    <div class='metric-value'>{probas[0][0]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-box' style='background:linear-gradient(135deg, #7c2d12, #f97316); color:white;'>
                    <div class='metric-label'>🐱 Probabilidad Gato</div>
                    <div class='metric-value'>{probas[0][1]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.error("❌ No se pudo leer la imagen.")
else:
    st.error("❌ Se requieren al menos 3 imágenes válidas por clase.")

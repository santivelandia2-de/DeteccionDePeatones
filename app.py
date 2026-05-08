import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ============================================
# CONFIGURACIÓN
# ============================================

st.set_page_config(
    page_title="Detección de Peatones",
    layout="wide"
)

st.title("🚶 Detección de Peatones con YOLO")
st.write("Aplicación de Object Detection usando YOLO entrenado localmente")


# ============================================
# CARGAR MODELO
# ============================================

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Configuración")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

show_labels = st.sidebar.checkbox(
    "Mostrar labels",
    value=True
)

show_confidence = st.sidebar.checkbox(
    "Mostrar confianza",
    value=True
)


# ============================================
# TABS
# ============================================

tab1, tab2 = st.tabs([
    "🖼 Imagen",
    "🎥 Video"
])


# ============================================
# FUNCIÓN DIBUJAR DETECCIONES
# ============================================

def draw_detections(image, results):

    annotated = image.copy()

    boxes = results[0].boxes

    count = 0

    for box in boxes:

        conf = float(box.conf[0])

        if conf < confidence:
            continue

        count += 1

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # =========================
        # Bounding box
        # =========================

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # =========================
        # Texto
        # =========================

        label = "Pedestrian"

        if show_confidence:
            label += f" {conf:.2f}"

        if show_labels:

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return annotated, count


# ============================================
# TAB IMAGEN
# ============================================

with tab1:

    st.header("Detección en Imagen")

    uploaded_image = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        image_np = np.array(image)

        if image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(
                image_np,
                cv2.COLOR_RGBA2RGB
            )

        # =========================
        # Predicción
        # =========================

        results = model.predict(
            source=image_np,
            conf=confidence,
            verbose=False
        )

        annotated_image, detections = draw_detections(
            image_np,
            results
        )

        # =========================
        # Layout
        # =========================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Imagen Original")

            st.image(
                image_np,
                use_container_width=True
            )

        with col2:

            st.subheader("Detecciones")

            st.image(
                annotated_image,
                use_container_width=True
            )

        st.success(
            f"Peatones detectados: {detections}"
        )


# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown(
    """
    ### Proyecto Deep Learning
    
    - YOLO Object Detection
    - Detección de peatones
    - Streamlit App
    """
)
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="DR Detection", page_icon="👁️", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #f9fafc;
        }
        h1, h2, h3, h4 {
            color: #1f3b73;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #1f3b73;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-size: 1em;
            font-weight: bold;
        }
        .prediction-box {
            padding: 20px;
            margin-top: 20px;
            background-color: #eaf4ff;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #1f3b73;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            font-size: 0.8em;
            color: #999;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
model = tf.keras.models.load_model("aptos2019_dr_model.keras")
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# --- TITLE ---
st.title("👁️ Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to detect the stage of diabetic retinopathy.")

# --- TABS ---
tab1, tab2 = st.tabs(["📤 Upload & Predict", "ℹ️ About"])

with tab1:
    uploaded_file = st.file_uploader("📂 Upload Retinal Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.write("### 🔄 Preprocessing Steps")
            st.write("- Resize to 224x224")
            st.write("- Normalize to [0,1]")
            st.write("- Expand dims for model input")

        # Preprocess
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        prob = np.max(prediction) * 100

        # --- Prediction Box ---
        st.markdown(
            f"<div class='prediction-box'>Prediction: {classes[predicted_class]}<br>Confidence: {prob:.2f}%</div>",
            unsafe_allow_html=True
        )

with tab2:
    st.write("### ℹ️ About This App")
    st.write("""
    - **Purpose**: Detect diabetic retinopathy stage from retinal fundus images.
    - **Model**: EfficientNetB0 pretrained on ImageNet, fine-tuned on APTOS 2019 dataset.
    - **Classes**:
        - No DR  
        - Mild  
        - Moderate  
        - Severe  
        - Proliferative DR  
    - **How it works**: Upload an image → Preprocessing → Model Prediction → Confidence Score.
    """)
    st.write("👩‍⚕️ *This tool is for educational purposes.*")

st.markdown("<div class='footer'></div>", unsafe_allow_html=True)

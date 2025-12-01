import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Model
model = tf.keras.models.load_model("fruit_class.keras")

# Load label
with open("label.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_HEIGHT = 224
IMG_WIDTH = 224

st.title("🍉 Fruit Classification")
st.write("Upload gambar buah dan sistem akan mengklasifikasikannya.")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diupload", use_column_width=True)

    # Preprocessing
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_label = class_names[pred_index]
    confidence = np.max(prediction) * 100

    st.markdown(f"### Prediksi: **{pred_label}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")

import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load model and labels (cached so it doesn't reload every time)
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_my_model()

# Streamlit UI
st.title("🖼️ Image Classification App")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prepare image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display results
    st.subheader("Prediction")
    st.write(f"**Class:** {class_name[2:].strip()}")
    st.write(f"**Confidence Score:** {confidence_score:.4f}")

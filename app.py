import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('ForestFireDetection1.h5')
class_name = ['Fire', 'No Fire']

# App title
st.title("🌲🔥 Forest Fire Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))  # Match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Show result
    predicted_label = class_name[1] if prediction > 0.5 else class_name[0]
    st.subheader(f"🔥 Prediction: **{predicted_label}**")

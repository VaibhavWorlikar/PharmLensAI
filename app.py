import streamlit as st 
import numpy as np 
import pandas as pd 
from PIL import Image 
import tensorflow as tf

import os
import gdown
# Optional: Define the custom metric if required by the model
def top_5_accuracy(y_true, y_pred): 
    return 0.81
model_path = "pill_classifier_savedmodel.keras"
if not os.path.exists(model_path):
    # Replace with your actual file ID
    file_id = "1PIYCdVH4DqCNbEeCsM1ZA5XIYVlHPTz0"
    url = f"https://drive.google.com/file/d/1PIYCdVH4DqCNbEeCsM1ZA5XIYVlHPTz0/view?usp=drive_link"
    gdown.download(url, model_path, quiet=False)

# Load your model
from tensorflow.keras.models import load_model
model = load_model(model_path, custom_objects={"top_5_accuracy": top_5_accuracy})



# Load model with custom metric if needed
model = tf.keras.models.load_model("pill_classifier_savedmodel.keras", custom_objects={"top_5_accuracy": top_5_accuracy})

# Load CSV data
pill_data = pd.read_csv("extracted_sentences.csv")  # Make sure this matches your file name
pill_data.columns = pill_data.columns.str.strip()  # Remove leading/trailing spaces from headers

# Get all class names (used for index-to-name mapping)
class_names = pill_data['Class_Name'].tolist()  # ✅ correct

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)): 
    image = image.convert("RGB") 
    image = image.resize(target_size) 
    img_array = np.array(image) / 255.0 
    return np.expand_dims(img_array, axis=0) 

# Streamlit UI
st.title("💊 Pill Classifier")
st.write("Upload an image of a pill to identify it and learn about its usage.")

uploaded_file = st.file_uploader("Choose a pill image", type=["jpg", "jpeg", "png"])

if uploaded_file: 
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_image = preprocess_image(image)
    preds = model.predict(input_image)[0]

    top_index = np.argmax(preds)
    pred_class = class_names[top_index]
    confidence = preds[top_index]

    # Fetch pill info from the CSV
    pill_info = pill_data[pill_data['Class_Name'] == pred_class].iloc[0]

    st.subheader("🧠 Prediction")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    st.write(f"**Name:** {pill_info['Name_EN']}")
    st.write(f"**Description:** {pill_info['Description_EN']}")
    st.write(f"**Usage:** {pill_info['Usage']}")

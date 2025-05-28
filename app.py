import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Load CSV with pill info
pill_data = pd.read_csv("extracted_sentences.csv")
pill_data.columns = pill_data.columns.str.strip()
class_names = pill_data['Class_Name'].tolist()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="pill_classifier_final.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_tflite(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    top_idx = np.argmax(output_data)
    confidence = output_data[top_idx]
    top_5_idx = np.argsort(output_data)[-5:][::-1]
    top_5_predictions = [(class_names[i], output_data[i]) for i in top_5_idx]

    return class_names[top_idx], confidence, top_5_predictions

# Streamlit UI
st.title("💊 PharmLensAI Pill Classifier ")
st.write("Upload a pill image to identify it and learn about its use.")

uploaded_file = st.file_uploader("Choose a pill image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    pred_class, confidence, top5 = predict_tflite(image)
    pill_info = pill_data[pill_data['Class_Name'] == pred_class].iloc[0]

    st.subheader("🧠 Prediction")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
    st.write(f"**Name:** {pill_info['Name_EN']}")
    st.write(f"**Description:** {pill_info['Description_EN']}")
    st.write(f"**Usage:** {pill_info['Usage']}")

    st.markdown("### 🔝 Top 5 Predictions:")
    for name, prob in top5:
        st.write(f"- {name}: {prob*100:.2f}%")

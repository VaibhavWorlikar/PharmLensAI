# PharmLensAI — Intelligent Pill Identification & Information System

**PharmLensAI** is an AI-powered pill classification web app that helps users identify pills from images and provides detailed information about their name, description, and usage. Built with TensorFlow and Streamlit, it leverages a deep learning model trained to classify a wide variety of pharmaceutical pills, making it easy to quickly recognize medications using just a photo.

---

## Features

* Upload a pill image and get instant classification results
* View top predicted pill classes with confidence scores
* Access detailed pill information such as name, description, and usage
* User-friendly interface built with Streamlit for easy accessibility
* Lightweight model optimized with TensorFlow Lite for efficient deployment

---

## Demo

*under development*

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/PharmLensAI.git
   cd PharmLensAI
   ```

2. Create and activate a Python virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open your browser and go to [http://localhost:8501](http://localhost:8501) to interact with PharmLensAI.

---

## Model Details

* Model file: `pill_classifier_final.tflite` (TensorFlow Lite format for efficient inference)
* Input image size: 224x224 pixels
* Model trained on a diverse dataset of pharmaceutical pill images
* Outputs top 5 predictions with confidence scores

---

## Project Structure

```
PharmLensAI/
│
├── app.py                   # Main Streamlit app code
├── pill_classifier_final.tflite  # TensorFlow Lite model file
├── extracted_sentences.csv  # CSV containing pill names, descriptions, and usage info
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License.

---

## Contact

For any questions or support, please contact:

* Vaibhav Worlikar : [vaibhavworlikar2004@gmail.com]
* GitHub: [https://github.com/VaibhavWorlikar]

---

Let me know if you want me to tailor it for a specific platform or add any other section!

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import zipfile

app = Flask(__name__, template_folder="templates")  # Ensure templates are loaded

MODEL_ZIP_PATH = "model.zip"
MODEL_PKL_PATH = "model.pkl"
VECTORIZER_PKL_PATH = "vectorizer.pkl"

# Extract model.zip if necessary
if not os.path.exists(MODEL_PKL_PATH) or not os.path.exists(VECTORIZER_PKL_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall()  # Extracts files in the same directory
    print("Model and Vectorizer extracted successfully!")

# Load vectorizer and model
def load_vectorizer():
    with open(VECTORIZER_PKL_PATH, "rb") as vec_file:
        return pickle.load(vec_file)

def load_model():
    with open(MODEL_PKL_PATH, "rb") as model_file:
        return pickle.load(model_file)

vectorizer = load_vectorizer()
model = load_model()
print("Model and Vectorizer loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sms_text = data.get("text", "").strip()

        if not sms_text:
            return jsonify({"error": "No text provided"}), 400

        transformed_text = vectorizer.transform([sms_text]).toarray() 
        prediction = model.predict(transformed_text)

        result = "Spam" if prediction[0] == 1 else "Not Spam"
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import subprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
model = load_model("fake_image_detector.h5")

# Extract metadata using exiftool
def get_metadata(image_path):
    command = f'exiftool "{image_path}"'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = {
        line.split(": ", 1)[0].strip(): line.split(": ", 1)[1].strip()
        for line in result.stdout.strip().split("\n") if ": " in line
    }
    return metadata

# Predict using trained model
def model_predict(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array, verbose=0)[0][0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        metadata = get_metadata(filepath)
        prediction = model_predict(filepath)

        x_resolution = metadata.get("X Resolution", "Unknown")
        megapixels = metadata.get("Megapixels", "0")

        try:
            megapixels = float(megapixels) if megapixels.replace('.', '', 1).isdigit() else 0
        except ValueError:
            megapixels = 0

        metadata_score = 0
        if x_resolution == "72":
            metadata_score += 0.3
        if megapixels >= 5:
            metadata_score += 0.3

        confidence = max(metadata_score, prediction)

        if prediction > 0.5 and metadata_score >= 0.5:
            label = "Image is Real"
        elif prediction > 0.5 and metadata_score < 0.5:
            label = "Image is Real But Shared or Manipulated"
        elif prediction < 0.5 and metadata_score < 0.5:
            label = "Image is Fake"
        elif prediction < 0.5 and metadata_score >= 0.5:
            label = "Image is Real"

        return render_template("result.html", filename=file.filename, label=label, confidence=f"{confidence:.2%}", raw_confidence=confidence * 100, x_res=x_resolution, mega=megapixels, metadata=metadata)

    return render_template("index.html")

@app.route('/metadata', methods=['GET', 'POST'])
def metadata_view():
    metadata = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return "No file selected", 400
        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        metadata = get_metadata(filepath)
        filename = file.filename
    return render_template("metadata.html", metadata=metadata, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

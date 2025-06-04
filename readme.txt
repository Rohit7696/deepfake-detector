# Deepfake Image Detection Web App

This is a Flask-based web application that allows users to upload an image and detect whether it is real or fake using a deep learning model. It also includes metadata analysis and a confidence graph visualization.

---

## 🚀 Features
- Upload image through a clean UI
- Predict if an image is real or fake using a trained model (`fake_image_detector.h5`)
- View prediction confidence on a line graph
- Analyze image metadata (e.g., resolution, megapixels)
- Separate pages for detection and metadata viewing

---

## 🛠 Tech Stack
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript (Chart.js)
- **Model:** TensorFlow / Keras
- **Preprocessing:** EfficientNet-based normalization
- **Metadata:** Extracted using `exiftool`

---

## 📂 Project Structure
```
├── server.py                  # Main Flask app
├── fake_image_detector.h5     # Trained deepfake detection model
├── templates/
│   ├── index.html             # Image upload page
│   ├── result.html            # Prediction + Graph UI
│   └── metadata.html          # Metadata viewing page
├── static/
│   └── styles.css             # UI styling
├── test_predict.py            # CLI script for standalone prediction
├── requirements.txt           # Python dependencies
├── assets/
│   ├── real_detect.png        # Screenshot showing real prediction
│   └── fake_detect.png        # Screenshot showing fake prediction
└── README.md
```

---

## 🖼️ Demo Screenshots

### ✅ Real Image Detection Result
![Real Image](assets/real_detect.png)

### ❌ Fake Image Detection Result
![Fake Image](assets/fake_detect.png)

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
```

2. **Set up a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install `exiftool` for metadata extraction:**
- Linux:
```bash
sudo apt install libimage-exiftool-perl
```
- macOS:
```bash
brew install exiftool
```
- Windows: [Download here](https://exiftool.org/)

5. **Run the application:**
```bash
python server.py
```

6. **Use the CLI tool (optional):**
```bash
python test_predict.py path/to/image.jpg
```

---

## 📊 Prediction Logic
- Combines model confidence and metadata score
- If both are high → likely real
- If both are low → likely fake
- Intermediate cases handled by thresholds

---

## ✅ Example Output
- Prediction: **Image is Fake**
- Confidence: **83.45%**
- X Resolution: **72**, Megapixels: **4.2**

---

## 📌 Notes
- The model used is a binary classifier (real vs. fake)
- The metadata score is a heuristic based on EXIF data

---

## 📃 License
This project is for educational and research purposes only.

---

## 👨‍💻 Author
**Rohit Pawar**  
For any questions or feedback, feel free to reach out.

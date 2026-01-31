from flask import Flask, render_template, request, jsonify
import os, json
from predict import predict_image
from agentic_doctor import agentic_doctor_response

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open("disease_info.json") as f:
    disease_info = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    disease, confidence = predict_image(path)
    confidence = round(confidence, 2)

    info = disease_info[disease]
    doctor = agentic_doctor_response(disease, confidence, info)

    # message depends on confidence, DATA DOES NOT
    if confidence < 0.40:
        message = f"POSSIBLE {disease.upper()} (LOW CONFIDENCE)"
    else:
        message = f"THIS IS MOST LIKELY {disease.upper()}"

    return jsonify({
        "message": message,
        "confidence": confidence,
        "severity_score": doctor["severity_score"],
        "severity_level": doctor["severity_level"],
        "symptoms": info["symptoms"],
        "description": info["description"],
        "cure": info["cure"],
        "consult_doctor_when": info["consult_doctor_when"],
        "agentic_doctor": doctor
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)


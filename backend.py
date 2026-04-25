from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)
CORS(app)

class_names = open("labels.txt", "r").readlines()


def calculate_model(model_type, image_bytes):

    np.set_printoptions(suppress=True)

    model = load_model(model_type, compile=False)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = int(np.argmax(prediction[0]))

    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    result1, conf1 = calculate_model("keras_Model.h5", image_bytes)
    result2, conf2 = calculate_model("balanced_model.h5", image_bytes)

    score = 0
    if "PNEUMONIA" in result1.upper():
        score += 1
    if "PNEUMONIA" in result2.upper():
        score += 1

    return jsonify({
        "model_1": result1,
        "confidence_1": conf1,
        "model_2": result2,
        "confidence_2": conf2,
        "final_score": score,
        "summary": f"{score}/2 models believe this person has pneumonia"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
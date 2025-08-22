from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder=".", static_url_path="")

# Load trained pipeline (model + preprocessing)
model = joblib.load("diabetes_model.pkl")

# Expected features in the same order as training
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

@app.route("/")
def home():
    # Serve your HTML frontend
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        row = [data.get(feat, None) for feat in FEATURES]
        X_input = np.array([row], dtype=float)
        proba = model.predict_proba(X_input)[0, 1]
        pred = int(proba >= 0.5)
        return jsonify({
            "probability_diabetes": float(proba),
            "prediction": pred,
            "threshold": 0.5
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

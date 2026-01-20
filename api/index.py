from flask import Flask, render_template, request
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        features = [
            float(request.form["fixed_acidity"]),
            float(request.form["volatile_acidity"]),
            float(request.form["citric_acid"]),
            float(request.form["residual_sugar"]),
            float(request.form["chlorides"]),
            float(request.form["free_sulfur_dioxide"]),
            float(request.form["total_sulfur_dioxide"]),
            float(request.form["density"]),
            float(request.form["pH"]),
            float(request.form["sulphates"]),
            float(request.form["alcohol"])
        ]

        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        confidence = round(max(model.predict_proba(features)[0]) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

# 👇 REQUIRED for Vercel
def handler(environ, start_response):
    return app(environ, start_response)

import os

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

model = joblib.load(os.path.join(os.path.dirname(__file__), "../model.pkl"))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), "../scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
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

# Vercel needs this
app = app

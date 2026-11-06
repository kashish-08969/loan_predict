from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Always resolve files relative to this app.py, not the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.pkl"

app = Flask(__name__)

NUMERIC = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CATEGORICAL = ["education", "self_employed"]
FEATURES = NUMERIC + CATEGORICAL


def load_model():
    """Load the trained model only when a prediction is requested."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Copy loan_model.pkl into the same folder as app.py."
        )
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "The model file exists but could not be opened. "
            "Check that it is the model downloaded from your training notebook "
            "and that your local scikit-learn version matches the training version."
        ) from exc


@app.route("/")
def home():
    # The home page can open even before the model file is copied.
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Please submit the form again."), 400

    try:
        row = {}
        for key in NUMERIC:
            value = float(data[key])
            if value < 0:
                return jsonify(error=f"{key.replace('_', ' ').title()} cannot be negative."), 400
            row[key] = value

        if not 300 <= row["cibil_score"] <= 900:
            return jsonify(error="CIBIL score must be between 300 and 900."), 400
        if row["loan_term"] <= 0:
            return jsonify(error="Loan term must be greater than zero."), 400

        for key in CATEGORICAL:
            value = str(data[key]).strip()
            if not value:
                return jsonify(error=f"Please select {key.replace('_', ' ')}."), 400
            row[key] = value

    except (KeyError, TypeError, ValueError):
        return jsonify(error="Please fill in every field with valid values."), 400

    try:
        model = load_model()
        input_df = pd.DataFrame([row], columns=FEATURES)
        probabilities = model.predict_proba(input_df)[0]

        # The original interface treats class 1 as approved. Use the model's
        # class labels when available, otherwise retain that convention.
        classes = list(getattr(model, "classes_", []))
        approved_index = classes.index(1) if 1 in classes else 1
        if approved_index >= len(probabilities):
            approved_index = len(probabilities) - 1

        probability = float(probabilities[approved_index]) * 100
        return jsonify(
            approved=probability >= 50,
            probability=round(probability, 1),
        )
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 500
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(error=f"Prediction failed: {type(exc).__name__}: {exc}"), 500


if __name__ == "__main__":
    # debug=False avoids the development reloader starting the app twice.
    app.run(host="127.0.0.1", port=5000, debug=False)

"""
app.py
======
Flask REST API for flight fare prediction (Stretch Challenge — Phase 7).

Endpoints:
    POST /predict  — Accept flight details as JSON, return predicted fare.
    GET  /health   — Simple health check.

Usage:
    docker-compose --profile api up api
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"airline":"Biman","source":"Dhaka","destination":"Chittagong","date":"2026-03-15"}'
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from flask import Flask, request, jsonify

from src.models import load_model
from src.feature_engineering import (
    create_date_features,
    encode_categoricals,
    load_training_columns,
    align_features,
    load_scaler,
)

app = Flask(__name__)

MODEL_PATH = Path("models/best_model.joblib")
TRAIN_COLUMNS_PATH = Path("data/processed/train_columns.json")

model = None
training_columns = None
scaler = None
scaler_columns = None


def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model


def get_training_columns():
    global training_columns
    if training_columns is None:
        training_columns = load_training_columns(TRAIN_COLUMNS_PATH)
    return training_columns


def get_scaler():
    global scaler, scaler_columns
    if scaler is None:
        scaler, scaler_columns = load_scaler()
    return scaler, scaler_columns


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON with keys matching the raw dataset columns.
    Returns {"predicted_fare": <float>}.
    """
    data = request.get_json(force=True)

    required_fields = ["airline", "source", "destination", "date"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = pd.DataFrame([{
            "Airline": data["airline"],
            "Source": data["source"],
            "Destination": data["destination"],
            "Date": data["date"],
            "Base Fare": data.get("base_fare", 0),
            "Tax & Surcharge": data.get("tax_surcharge", 0),
        }])

        row = create_date_features(row)
        row = encode_categoricals(row)

        non_numeric = row.select_dtypes(exclude="number").columns.tolist()
        row = row.drop(columns=non_numeric, errors="ignore")

        fitted_scaler, scale_cols = get_scaler()
        present = [c for c in scale_cols if c in row.columns]
        if present:
            row[present] = fitted_scaler.transform(row[present])

        row = align_features(row, get_training_columns())

        m = get_model()
        prediction = m.predict(row)[0]

        return jsonify({"predicted_fare": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

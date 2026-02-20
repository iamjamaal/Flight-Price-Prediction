"""
app.py
======
Flask REST API for flight fare prediction (Stretch Challenge — Phase 8).

Endpoints:
    POST /predict  — Accept flight details as JSON, return predicted fare in BDT.
    GET  /health   — Simple health check.

Usage:
    docker compose --profile api up api
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"airline":"Biman Bangladesh Airlines","source":"DAC","destination":"CGP","date":"2026-03-15"}'
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

from src.models import load_model
from src.feature_engineering import (
    load_training_columns,
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


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Flight Fare Prediction API",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /predict": "Predict fare — send JSON with airline, source, destination, date",
        },
        "example": {
            "url": "POST /predict",
            "body": {
                "airline": "Biman Bangladesh Airlines",
                "source": "DAC",
                "destination": "CGP",
                "date": "2026-03-15",
                "class": "Economy",
            }
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON with keys matching the raw dataset columns.
    Returns {"predicted_fare_bdt": <float>}.
    """
    data = request.get_json(force=True)

    required_fields = ["airline", "source", "destination", "date"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # Parse date and derive temporal features
        dt = pd.to_datetime(data["date"])
        month = dt.month
        season_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Summer",  4: "Summer",  5: "Summer",
            6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
        season = season_map[month]
        weekday_name = dt.day_name()

        airline    = data["airline"]
        source     = data["source"]
        destination = data["destination"]
        stopovers  = data.get("stopovers", "Non-stop")
        travel_class = data.get("class", "Economy")
        aircraft_type = data.get("aircraft_type", "Boeing 737")
        booking_source = data.get("booking_source", "Online Website")
        duration   = float(data.get("duration", 2.0))
        days_before = int(data.get("days_before_departure", 30))

        # Build a zero-vector aligned to training columns, then set features
        training_columns = get_training_columns()
        row = pd.DataFrame([[0] * len(training_columns)], columns=training_columns)

        # Numeric features
        for col, val in [
            ("Duration", duration),
            ("DaysBeforeDeparture", days_before),
            ("Month", month),
            ("Day", dt.day),
            ("Weekday", dt.weekday()),
        ]:
            if col in row.columns:
                row[col] = val

        # One-hot features — set to 1 if the column exists in training schema
        one_hot = [
            f"Airline_{airline}",
            f"Source_{source}",
            f"Destination_{destination}",
            f"Stopovers_{stopovers}",
            f"Aircraft Type_{aircraft_type}",
            f"Class_{travel_class}",
            f"Booking Source_{booking_source}",
            f"Season_{season}",
            f"WeekdayName_{weekday_name}",
        ]
        for col in one_hot:
            if col in row.columns:
                row[col] = 1

        m = get_model()
        pred_log = m.predict(row)[0]
        prediction = float(np.expm1(pred_log))

        return jsonify({"predicted_fare_bdt": round(max(prediction, 0), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

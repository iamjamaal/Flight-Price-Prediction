"""
app.py
Flask REST API for flight fare prediction (Stretch Challenge — Phase 8).

Endpoints:
    GET  /         — Service info and usage example
    GET  /health   — Simple health check
    POST /predict  — Accept flight details as JSON, return predicted fare in BDT

Swagger UI: http://localhost:5000/apidocs/

Usage:
    docker compose --profile api up api
    curl -X POST http://localhost:5000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"airline":"Biman Bangladesh Airlines","source":"DAC","destination":"CGP","date":"2026-03-15"}'
"""


import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger

from src.models import load_model
from src.feature_engineering import load_training_columns, load_scaler
from src.constants import SEASON_MAP

app = Flask(__name__)

# Rate limiting (in-memory; no Redis required)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

# Swagger
swagger = Swagger(
    app,
    config={
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/",
    },
    template={
        "info": {
            "title": "Flight Fare Prediction API",
            "description": "REST API for predicting domestic flight fares in Bangladesh.",
            "version": "1.0.0",
        },
        "schemes": ["http", "https"],
    },
)


# Paths 
MODEL_PATH = Path("models/best_model.joblib")
TRAIN_COLUMNS_PATH = Path("data/processed/train_columns.json")
KNOWN_VALUES_PATH = Path("data/processed/known_values.json")

# Lazy-loaded globals
model = None
training_columns = None
scaler = None
scaler_columns = None
_known_values = None


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


def get_known_values():
    """
    Lazily load known categorical values for input validation.
    Returns None if the file is absent (graceful degradation — validation skipped).
    """
    global _known_values
    if _known_values is None:
        if KNOWN_VALUES_PATH.exists():
            _known_values = json.loads(KNOWN_VALUES_PATH.read_text(encoding="utf-8"))
        else:
            _known_values = {}
    return _known_values or None


# Routes 

@app.route("/", methods=["GET"])
def index():
    """
    Service info and usage example.
    ---
    tags:
      - Info
    responses:
      200:
        description: Service description with endpoint listing and usage example.
        schema:
          type: object
          properties:
            service:
              type: string
              example: Flight Fare Prediction API
            endpoints:
              type: object
            example:
              type: object
    """
    return jsonify({
        "service": "Flight Fare Prediction API",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /predict": "Predict fare — send JSON with airline, source, destination, date",
            "GET  /apidocs/": "Interactive Swagger UI",
        },
        "example": {
            "url": "POST /predict",
            "body": {
                "airline": "Biman Bangladesh Airlines",
                "source": "DAC",
                "destination": "CGP",
                "date": "2026-03-15",
                "class": "Economy",
            },
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check.
    ---
    tags:
      - Health
    responses:
      200:
        description: Service is healthy.
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
    """
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
@limiter.limit("30 per minute")
def predict():
    """
    Predict flight fare.
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - airline
            - source
            - destination
            - date
          properties:
            airline:
              type: string
              example: Biman Bangladesh Airlines
            source:
              type: string
              example: DAC
            destination:
              type: string
              example: CGP
            date:
              type: string
              format: date
              example: "2026-03-15"
            class:
              type: string
              example: Economy
              default: Economy
            stopovers:
              type: string
              example: Non-stop
              default: Non-stop
            duration:
              type: number
              example: 2.0
              default: 2.0
            days_before_departure:
              type: integer
              example: 30
              default: 30
            aircraft_type:
              type: string
              example: Boeing 737
            booking_source:
              type: string
              example: Online Website
    responses:
      200:
        description: Predicted fare in BDT.
        schema:
          type: object
          properties:
            predicted_fare_bdt:
              type: number
              example: 8500.0
      400:
        description: Bad request — missing fields, wrong content-type, or unknown categorical value.
        schema:
          type: object
          properties:
            error:
              type: string
            accepted_values:
              type: array
              items:
                type: string
      500:
        description: Internal server error.
        schema:
          type: object
          properties:
            error:
              type: string
    """
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    required_fields = ["airline", "source", "destination", "date"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Input validation against known training values (skipped if file absent)
    known = get_known_values()
    if known:
        for field, key in [
            ("airline", "Airline"),
            ("source", "Source"),
            ("destination", "Destination"),
            ("stopovers", "Stopovers"),
            ("class", "Class"),
        ]:
            if key in known and field in data:
                if data[field] not in known[key]:
                    return jsonify({
                        "error": f"Unknown {field}: '{data[field]}'",
                        "accepted_values": known[key],
                    }), 400

    try:
        dt = pd.to_datetime(data["date"])
        month = dt.month
        season = SEASON_MAP[month]
        weekday_name = dt.day_name()

        airline = data["airline"]
        source = data["source"]
        destination = data["destination"]
        stopovers = data.get("stopovers", "Non-stop")
        travel_class = data.get("class", "Economy")
        aircraft_type = data.get("aircraft_type", "Boeing 737")
        booking_source = data.get("booking_source", "Online Website")
        duration = float(data.get("duration", 2.0))
        days_before = int(data.get("days_before_departure", 30))

        training_cols = get_training_columns()
        row = pd.DataFrame([[0] * len(training_cols)], columns=training_cols)

        for col, val in [
            ("Duration", duration),
            ("DaysBeforeDeparture", days_before),
            ("Month", month),
            ("Day", dt.day),
            ("Weekday", dt.weekday()),
        ]:
            if col in row.columns:
                row[col] = val

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
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)

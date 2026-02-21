"""
tests/test_api.py
==================
Unit tests for app/app.py — all model and file I/O is mocked.
No .joblib or data files are required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

_TEST_COLUMNS = [
    "Duration",
    "DaysBeforeDeparture",
    "Month",
    "Day",
    "Weekday",
    "Airline_Biman Bangladesh Airlines",
    "Source_DAC",
    "Destination_CGP",
    "Season_Winter",
    "Class_Economy",
]

_KNOWN_VALUES = {
    "Airline": ["Biman Bangladesh Airlines", "US-Bangla Airlines"],
    "Source": ["DAC", "CGP"],
    "Destination": ["DAC", "CGP"],
    "Stopovers": ["Non-stop", "1 Stop"],
    "Class": ["Economy", "Business"],
}

_VALID_PAYLOAD = {
    "airline": "Biman Bangladesh Airlines",
    "source": "DAC",
    "destination": "CGP",
    "date": "2026-03-15",
}


def _make_mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([np.log1p(5000)])
    return m


@pytest.fixture
def client():
    """Flask test client with all I/O mocked; rate limiting disabled."""
    from app.app import app

    mock_model = _make_mock_model()

    with (
        patch("app.app.get_model", return_value=mock_model),
        patch("app.app.get_training_columns", return_value=_TEST_COLUMNS),
        patch("app.app.get_scaler", return_value=(MagicMock(), [])),
        patch("app.app.get_known_values", return_value=None),
    ):
        app.config["TESTING"] = True
        app.config["RATELIMIT_ENABLED"] = False
        with app.test_client() as c:
            yield c


@pytest.fixture
def known_values_client():
    """Flask test client with known_values populated for validation tests."""
    from app.app import app

    mock_model = _make_mock_model()

    with (
        patch("app.app.get_model", return_value=mock_model),
        patch("app.app.get_training_columns", return_value=_TEST_COLUMNS),
        patch("app.app.get_scaler", return_value=(MagicMock(), [])),
        patch("app.app.get_known_values", return_value=_KNOWN_VALUES),
    ):
        app.config["TESTING"] = True
        app.config["RATELIMIT_ENABLED"] = False
        with app.test_client() as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIndexEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_has_service_key(self, client):
        data = client.get("/").get_json()
        assert "service" in data

    def test_has_endpoints_key(self, client):
        data = client.get("/").get_json()
        assert "endpoints" in data


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body_is_ok(self, client):
        data = client.get("/health").get_json()
        assert data == {"status": "ok"}


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        resp = client.post("/predict", json=_VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_valid_request_has_predicted_fare_key(self, client):
        data = client.post("/predict", json=_VALID_PAYLOAD).get_json()
        assert "predicted_fare_bdt" in data

    def test_predicted_fare_is_non_negative(self, client):
        data = client.post("/predict", json=_VALID_PAYLOAD).get_json()
        assert data["predicted_fare_bdt"] >= 0

    def test_missing_required_field_returns_400(self, client):
        payload = {"airline": "Biman Bangladesh Airlines", "source": "DAC"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400

    def test_wrong_content_type_returns_400(self, client):
        resp = client.post(
            "/predict", data="not json", content_type="text/plain"
        )
        assert resp.status_code == 400

    def test_error_response_has_error_key(self, client):
        payload = {"airline": "X"}  # missing required fields
        data = client.post("/predict", json=payload).get_json()
        assert "error" in data

    def test_unknown_airline_with_validation_returns_400(self, known_values_client):
        payload = {**_VALID_PAYLOAD, "airline": "Unknown Airline XYZ"}
        resp = known_values_client.post("/predict", json=payload)
        assert resp.status_code == 400

    def test_unknown_airline_error_includes_accepted_values(self, known_values_client):
        payload = {**_VALID_PAYLOAD, "airline": "Unknown Airline XYZ"}
        data = known_values_client.post("/predict", json=payload).get_json()
        assert "accepted_values" in data

    def test_valid_airline_with_validation_returns_200(self, known_values_client):
        resp = known_values_client.post("/predict", json=_VALID_PAYLOAD)
        assert resp.status_code == 200

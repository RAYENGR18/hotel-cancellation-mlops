"""Tests de l'API avec un faux modèle (pas besoin de MLflow)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model import load_model

# Une réservation valide, réutilisée dans tous les tests.
BOOKING = {
    "hotel": "City Hotel",
    "lead_time": 120,
    "adr": 95.0,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "previous_cancellations": 0,
    "days_in_waiting_list": 0,
    "total_of_special_requests": 1,
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "reserved_room_type": "A",
    "country": "PRT",
}


class FakeModel:
    """Remplace le vrai modèle : 0,9 si dépôt non remboursable, 0,2 sinon."""

    def predict_proba(self, df):
        p = np.where(df["deposit_type"] == "Non Refund", 0.9, 0.2)
        return np.column_stack([1 - p, p])


@pytest.fixture
def client():
    """Client de test avec le faux modèle chargé."""
    app.state.model = FakeModel()
    yield TestClient(app)
    app.state.model = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_low_risk(client):
    r = client.post("/predict", json=BOOKING)
    assert r.status_code == 200
    assert r.json() == {"probability": 0.2, "will_cancel": False}


def test_predict_high_risk(client):
    r = client.post("/predict", json={**BOOKING, "deposit_type": "Non Refund"})
    assert r.status_code == 200
    assert r.json()["will_cancel"] is True


def test_predict_batch(client):
    payload = {"bookings": [BOOKING, {**BOOKING, "deposit_type": "Non Refund"}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    assert [p["will_cancel"] for p in r.json()["predictions"]] == [False, True]


def test_negative_lead_time_rejected(client):
    r = client.post("/predict", json={**BOOKING, "lead_time": -3})
    assert r.status_code == 422


def test_unknown_hotel_rejected(client):
    r = client.post("/predict", json={**BOOKING, "hotel": "Hilton"})
    assert r.status_code == 422


def test_leakage_field_rejected(client):
    r = client.post("/predict", json={**BOOKING, "reservation_status": "Canceled"})
    assert r.status_code == 422


def test_empty_batch_rejected(client):
    r = client.post("/predict/batch", json={"bookings": []})
    assert r.status_code == 422


def test_no_model_returns_503(client):
    app.state.model = None
    assert client.post("/predict", json=BOOKING).status_code == 503


def test_load_model_without_uri(monkeypatch):
    monkeypatch.delenv("MODEL_URI", raising=False)
    assert load_model() is None


def test_load_model_bad_uri(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "models:/modele-inexistant@production")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns-test-vide")
    assert load_model() is None

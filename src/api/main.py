"""API REST de prédiction des annulations de réservations.

Lancement local :
    poetry run uvicorn src.api.main:app --reload
"""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.model import load_model
from src.api.schemas import BatchRequest, BatchResponse, Booking, Prediction

# Seuil de décision : au-dessus, la réservation est jugée « à risque ».
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé une seule fois, au démarrage du serveur.
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Hotel Cancellation API",
    description="Probabilité d'annulation d'une réservation hôtelière",
    version="0.1.0",
    lifespan=lifespan,
)


def _predict(request: Request, bookings: list[Booking]) -> list[Prediction]:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    df = pd.DataFrame([b.model_dump() for b in bookings])
    probas = model.predict_proba(df)[:, 1]
    return [
        Prediction(probability=float(p), will_cancel=bool(p >= THRESHOLD))
        for p in probas
    ]


@app.get("/health")
def health(request: Request) -> dict:
    """Vérifie que l'API répond et indique si un modèle est chargé."""
    model = getattr(request.app.state, "model", None)
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(booking: Booking, request: Request) -> Prediction:
    """Prédit le risque d'annulation d'une réservation."""
    return _predict(request, [booking])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    """Prédit le risque d'annulation de plusieurs réservations."""
    return BatchResponse(predictions=_predict(request, payload.bookings))

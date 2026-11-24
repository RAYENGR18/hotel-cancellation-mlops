"""Chargement du modèle depuis le Model Registry MLflow."""

import logging
import os

logger = logging.getLogger(__name__)


def load_model():
    """Charge le modèle indiqué par MODEL_URI, ou renvoie None.

    Exemple : MODEL_URI="models:/hotel-cancellation@production".
    Si le modèle est introuvable, l'API démarre quand même : /health le
    signale et /predict répond 503 au lieu de planter.
    """
    uri = os.getenv("MODEL_URI")
    if not uri:
        logger.warning("MODEL_URI non défini : l'API démarre sans modèle")
        return None
    try:
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(uri)
        logger.info("Modèle chargé : %s", uri)
        return model
    except Exception:
        logger.exception("Impossible de charger le modèle %s", uri)
        return None

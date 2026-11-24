"""Schémas Pydantic des requêtes et réponses de l'API.

Chaque requête est validée avant d'atteindre le modèle : un champ manquant,
une valeur négative ou une catégorie inconnue renvoie une erreur 422.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Booking(BaseModel):
    """Une réservation à évaluer (variables connues au moment de la réservation)."""

    # Refuse tout champ en trop, par exemple `reservation_status` (fuite de données).
    model_config = ConfigDict(extra="forbid")

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0, description="Jours entre réservation et arrivée")
    adr: float = Field(ge=0, description="Prix moyen par nuit (€)")
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: int = Field(ge=0)
    previous_cancellations: int = Field(ge=0)
    days_in_waiting_list: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)
    meal: Literal["BB", "HB", "FB", "SC", "Undefined"]
    market_segment: str
    distribution_channel: str
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"]
    reserved_room_type: str = Field(min_length=1, max_length=1)
    country: str = Field(min_length=2, max_length=3)


class Prediction(BaseModel):
    """Résultat pour une réservation."""

    probability: float = Field(description="Probabilité d'annulation (0 à 1)")
    will_cancel: bool = Field(description="True si la probabilité dépasse le seuil")


class BatchRequest(BaseModel):
    """Plusieurs réservations évaluées en un seul appel."""

    bookings: list[Booking] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]

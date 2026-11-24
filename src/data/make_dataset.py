"""Préparation des données : nettoyage puis découpage temporel.

Lit le fichier brut des réservations, retire les colonnes qui provoquent une
fuite de données, traite les valeurs manquantes et sépare les données en deux :
les arrivées 2015-2016 pour l'entraînement et les arrivées 2017 pour simuler
le flux de production.

Usage :
    python -m src.data.make_dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "is_canceled"

# Colonnes connues seulement APRÈS la réservation : les garder ferait tricher
# le modèle (fuite de données).
LEAKAGE_COLUMNS = [
    "reservation_status",  # contient directement la réponse (Canceled, Check-Out)
    "reservation_status_date",  # date de l'annulation ou du départ
    "assigned_room_type",  # chambre attribuée à l'arrivée
    "booking_changes",  # modifications faites après la réservation
]

# Colonnes retirées car presque vides (94 % de valeurs manquantes).
DROPPED_COLUMNS = ["company"]

TRAIN_YEARS = (2015, 2016)
PRODUCTION_YEAR = 2017

RAW_PATH = Path("data/raw/hotel_bookings.csv")
PROCESSED_DIR = Path("data/processed")


def load_raw(path: Path) -> pd.DataFrame:
    """Charge le fichier CSV brut des réservations."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {path}")
    df = pd.read_csv(path)
    logger.info("Données brutes : %d lignes, %d colonnes", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les réservations sans toucher à la variable cible.

    - retire les colonnes de fuite et les colonnes presque vides ;
    - remplace les valeurs manquantes de `children`, `country` et `agent` ;
    - supprime les réservations impossibles (aucun voyageur, prix négatif).
    """
    df = df.drop(columns=LEAKAGE_COLUMNS + DROPPED_COLUMNS, errors="ignore").copy()

    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("UNK")
    # agent = identifiant de l'agence ; 0 signifie « réservation sans agence ».
    df["agent"] = df["agent"].fillna(0).astype(int)

    total_guests = df["adults"] + df["children"] + df["babies"]
    valid = (total_guests > 0) & (df["adr"] >= 0)
    logger.info("Réservations invalides supprimées : %d", int((~valid).sum()))

    return df[valid].reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données selon l'année d'arrivée.

    Retourne (train, production) : 2015-2016 pour entraîner le modèle,
    2017 pour simuler les réservations qui arrivent en production.
    """
    year = df["arrival_date_year"]
    train = df[year.isin(TRAIN_YEARS)].reset_index(drop=True)
    production = df[year == PRODUCTION_YEAR].reset_index(drop=True)
    return train, production


def main(argv: list[str] | None = None) -> None:
    """Exécute la préparation complète et écrit les deux fichiers de sortie."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    train, production = temporal_split(clean(load_raw(args.raw)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_dir / "train.csv", index=False)
    production.to_csv(args.out_dir / "production.csv", index=False)

    for name, part in (("train", train), ("production", production)):
        logger.info(
            "%s : %d lignes, taux d'annulation %.1f %%",
            name,
            len(part),
            100 * part[TARGET].mean(),
        )


if __name__ == "__main__":
    main()

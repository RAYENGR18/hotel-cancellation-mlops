"""Tests unitaires de la préparation des données."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.make_dataset import LEAKAGE_COLUMNS, clean, main, temporal_split


@pytest.fixture
def raw_bookings() -> pd.DataFrame:
    """Petit jeu de réservations fictives avec les cas à nettoyer."""
    return pd.DataFrame(
        {
            "is_canceled": [0, 1, 0, 1, 0],
            "arrival_date_year": [2015, 2016, 2017, 2017, 2016],
            "adults": [2, 1, 2, 0, 2],
            "children": [0.0, None, 1.0, 0.0, 0.0],
            "babies": [0, 0, 0, 0, 0],
            "adr": [80.0, 95.5, 120.0, 60.0, -5.0],
            "country": ["PRT", None, "FRA", "ESP", "PRT"],
            "agent": [9.0, None, 240.0, 9.0, 1.0],
            "company": [None, None, 40.0, None, None],
            "reservation_status": [
                "Check-Out",
                "Canceled",
                "Check-Out",
                "Canceled",
                "Check-Out",
            ],
            "reservation_status_date": ["2015-07-03"] * 5,
            "assigned_room_type": ["A", "A", "D", "A", "A"],
            "booking_changes": [0, 1, 0, 0, 2],
        }
    )


def test_clean_removes_leakage_columns(raw_bookings: pd.DataFrame) -> None:
    cleaned = clean(raw_bookings)
    assert not set(LEAKAGE_COLUMNS) & set(cleaned.columns)
    assert "company" not in cleaned.columns
    assert "is_canceled" in cleaned.columns


def test_clean_fills_missing_values(raw_bookings: pd.DataFrame) -> None:
    cleaned = clean(raw_bookings)
    assert cleaned[["children", "country", "agent"]].isna().sum().sum() == 0
    assert "UNK" in cleaned["country"].values


def test_clean_removes_invalid_bookings(raw_bookings: pd.DataFrame) -> None:
    cleaned = clean(raw_bookings)
    # Une réservation sans voyageur et une avec un prix négatif sont retirées.
    assert len(cleaned) == 3
    assert (cleaned["adr"] >= 0).all()


def test_clean_does_not_modify_input(raw_bookings: pd.DataFrame) -> None:
    before = raw_bookings.copy()
    clean(raw_bookings)
    pd.testing.assert_frame_equal(raw_bookings, before)


def test_temporal_split_separates_years(raw_bookings: pd.DataFrame) -> None:
    train, production = temporal_split(clean(raw_bookings))
    assert set(train["arrival_date_year"]) <= {2015, 2016}
    assert set(production["arrival_date_year"]) == {2017}
    assert len(train) + len(production) == 3


def test_main_writes_both_files(raw_bookings: pd.DataFrame, tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    raw_bookings.to_csv(raw_path, index=False)
    out_dir = tmp_path / "processed"

    main(["--raw", str(raw_path), "--out-dir", str(out_dir)])

    assert (out_dir / "train.csv").exists()
    assert (out_dir / "production.csv").exists()


def test_main_fails_without_raw_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["--raw", str(tmp_path / "absent.csv")])

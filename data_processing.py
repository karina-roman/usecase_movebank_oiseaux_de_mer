# Chargement, nettoyage, filtrage et synthèses générales des données

import os
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg


def find_data_file() -> str:
    """Utilise le CSV local ou l'URL configurée sur Render."""

    candidate_paths = [
        cfg.APP_DIR / "data" / cfg.FILE_NAME,
        cfg.APP_DIR.parent / "data" / cfg.FILE_NAME,
        Path.cwd() / "data" / cfg.FILE_NAME,
    ]

    # En local, utiliser le CSV présent dans le projet
    data_path = next((path for path in candidate_paths if path.is_file()), None)

    if data_path is not None:
        return str(data_path)

    # Sur Render, utiliser le lien enregistré dans DATA_URL
    data_url = os.getenv("DATA_URL")

    if data_url:
        return data_url.strip()

    raise FileNotFoundError(
        f"Le fichier « {cfg.FILE_NAME} » est introuvable. Placez-le dans le dossier data/ ou définissez DATA_URL sur Render."
    )


@st.cache_data(show_spinner="Chargement et préparation des trajectoires…")
def load_and_prepare_data(path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Charge, nettoie et enrichit les positions avec les métriques de mouvement."""

    data = pd.read_csv(path, low_memory=False)
    raw_row_count = len(data)

    required_columns = {
        cfg.TIME_COLUMN,
        cfg.LAT_COLUMN,
        cfg.LON_COLUMN,
        cfg.ID_COLUMN,
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError("Colonnes obligatoires manquantes : " + ", ".join(sorted(missing_columns)))

    data[cfg.TIME_COLUMN] = pd.to_datetime(data[cfg.TIME_COLUMN], errors="coerce", utc=True)
    data[cfg.LAT_COLUMN] = pd.to_numeric(data[cfg.LAT_COLUMN], errors="coerce")
    data[cfg.LON_COLUMN] = pd.to_numeric(data[cfg.LON_COLUMN], errors="coerce")
    data[cfg.ID_COLUMN] = data[cfg.ID_COLUMN].astype("string").str.strip()

    valid_coordinates = (data[cfg.LAT_COLUMN].between(-90, 90) & data[cfg.LON_COLUMN].between(-180, 180))
    complete_rows = data[[cfg.TIME_COLUMN, cfg.LAT_COLUMN, cfg.LON_COLUMN, cfg.ID_COLUMN]].notna().all(axis=1)

    invalid_position_count = int((~(valid_coordinates & complete_rows)).sum())
    data = data.loc[valid_coordinates & complete_rows].copy()

    manual_outlier_count = 0
    if "manually-marked-outlier" in data.columns:
        marked_outlier = (
            data["manually-marked-outlier"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "oui"})
        )
        manual_outlier_count = int(marked_outlier.sum())
        data = data.loc[~marked_outlier].copy()

    if data.empty:
        raise ValueError("Aucune position valide après le nettoyage des données.")

    data = data.sort_values([cfg.ID_COLUMN, cfg.TIME_COLUMN]).reset_index(drop=True)

    # Movebank fournit ground-speed en m/s, l'application travaille en km/h.
    if "ground-speed" in data.columns:
        data["ground_speed_kmh"] = (pd.to_numeric(data["ground-speed"], errors="coerce") * 3.6)
    else:
        data["ground_speed_kmh"] = np.nan

    # Proxy conservé depuis la version initiale : première position valide.
    nest_positions = (
        data.groupby(cfg.ID_COLUMN, sort=False)[[cfg.LAT_COLUMN, cfg.LON_COLUMN]]
        .first()
        .rename(
            columns={
                cfg.LAT_COLUMN: "nest_lat",
                cfg.LON_COLUMN: "nest_lon",
            }
        )
    )
    data = data.join(nest_positions, on=cfg.ID_COLUMN)

    nest_lat_1 = np.radians(data["nest_lat"])
    nest_lat_2 = np.radians(data[cfg.LAT_COLUMN])
    nest_delta_lat = nest_lat_2 - nest_lat_1
    nest_delta_lon = np.radians(data[cfg.LON_COLUMN] - data["nest_lon"])
    nest_haversine_a = (
        np.sin(nest_delta_lat / 2) ** 2
        + np.cos(nest_lat_1)
        * np.cos(nest_lat_2)
        * np.sin(nest_delta_lon / 2) ** 2
    )
    data["distance_nest_km"] = (2 * cfg.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(nest_haversine_a, 0, 1))))

    grouped = data.groupby(cfg.ID_COLUMN, sort=False)
    previous_lat = grouped[cfg.LAT_COLUMN].shift()
    previous_lon = grouped[cfg.LON_COLUMN].shift()
    data["delta_time_hours"] = (grouped[cfg.TIME_COLUMN].diff().dt.total_seconds().div(3_600))

    lat_1 = np.radians(previous_lat)
    lat_2 = np.radians(data[cfg.LAT_COLUMN])
    delta_lat = lat_2 - lat_1
    delta_lon = np.radians(data[cfg.LON_COLUMN] - previous_lon)
    haversine_a = (np.sin(delta_lat / 2) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_lon / 2) ** 2)
    haversine_a = np.clip(haversine_a, 0, 1)

    data["segment_distance_km"] = (2 * cfg.EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine_a)))
    data["segment_speed_kmh"] = (data["segment_distance_km"] / data["delta_time_hours"]).replace([np.inf, -np.inf], np.nan)
    data["date"] = data[cfg.TIME_COLUMN].dt.tz_convert(None).dt.normalize()

    quality = {
        "raw_rows": raw_row_count,
        "invalid_positions": invalid_position_count,
        "manual_outliers": manual_outlier_count,
        "clean_rows": len(data),
    }
    return data, quality


def normalize_date_range(selected_dates, min_date: date, max_date: date) -> Tuple[date, date]:
    """Transforme la valeur de st.date_input en couple début/fin fiable."""

    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            return selected_dates[0], selected_dates[1]
        if len(selected_dates) == 1:
            return selected_dates[0], selected_dates[0]
        return min_date, max_date
    return selected_dates, selected_dates


def filter_trajectories(data: pd.DataFrame, selected_individuals: Iterable[str], start_date: date, end_date: date) -> Tuple[pd.DataFrame, int]:
    """Applique les filtres partagés et prépare les segments exploitables."""

    date_values = data[cfg.TIME_COLUMN].dt.date
    filtered = data.loc[data[cfg.ID_COLUMN].isin(selected_individuals) & date_values.between(start_date, end_date)].copy()

    if filtered.empty:
        return filtered, 0

    filtered["valid_segment"] = filtered["delta_time_hours"].gt(0)
    filtered["distance_km"] = filtered["segment_distance_km"].where(filtered["valid_segment"], 0.0)
    filtered["ground_speed_for_stats_kmh"] = filtered["ground_speed_kmh"]

    invalid_segment_count = int(((~filtered["valid_segment"]) & filtered["segment_distance_km"].notna()).sum())
    return filtered, invalid_segment_count


def build_individual_summary(filtered: pd.DataFrame, daily_distance: pd.DataFrame) -> pd.DataFrame:
    """Calcule le tableau de synthèse par individu."""

    summary = (
        filtered.groupby(cfg.ID_COLUMN, as_index=False, observed=True)
        .agg(
            positions=(cfg.TIME_COLUMN, "size"),
            first_position=(cfg.TIME_COLUMN, "min"),
            last_position=(cfg.TIME_COLUMN, "max"),
            total_distance_km=("distance_km", "sum"),
            mean_speed_kmh=("ground_speed_for_stats_kmh", "mean"),
            max_speed_kmh=("ground_speed_for_stats_kmh", "max"),
        )
        .sort_values(cfg.ID_COLUMN)
    )

    active_days = (daily_distance.groupby(cfg.ID_COLUMN, as_index=False, observed=True).agg(active_days=("date", "nunique"), mean_daily_distance_km=("distance_km", "mean"),))
    summary = summary.merge(active_days, on=cfg.ID_COLUMN, how="left")
    return summary[
        [
            cfg.ID_COLUMN,
            "positions",
            "active_days",
            "total_distance_km",
            "mean_daily_distance_km",
            "mean_speed_kmh",
            "max_speed_kmh",
            "first_position",
            "last_position",
        ]
    ]
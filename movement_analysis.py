# Calculs temporels et comportementaux liés aux déplacements

from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config as cfg


def build_daily_distance(filtered: pd.DataFrame) -> pd.DataFrame:
    """Calcule la distance parcourue par individu et par jour."""

    return (filtered.groupby([cfg.ID_COLUMN, "date"], as_index=False, observed=True)["distance_km"].sum().sort_values([cfg.ID_COLUMN, "date"]))


def build_hourly_summary(filtered: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construit les fenêtres temporelles et le cycle horaire médian."""

    valid_points = filtered.loc[filtered["valid_segment"]].copy()
    valid_points["window_start"] = valid_points[cfg.TIME_COLUMN].dt.floor(f"{cfg.WINDOW_MINUTES}min")

    windows = (
        valid_points.groupby([cfg.ID_COLUMN, "window_start"], as_index=False, observed=True)
        .agg(
            path_km=("distance_km", "sum"),
            n_positions=(cfg.TIME_COLUMN, "size"),
            first_position=(cfg.TIME_COLUMN, "min"),
            last_position=(cfg.TIME_COLUMN, "max"),
        )
    )
    windows["coverage_min"] = (windows["last_position"] - windows["first_position"]).dt.total_seconds().div(60)

    windows = windows.loc[windows["n_positions"].ge(cfg.MIN_WINDOW_POSITIONS) & windows["coverage_min"].ge(cfg.MIN_WINDOW_COVERAGE_MINUTES)].copy()

    hourly = (
        windows.assign(hour=windows["window_start"].dt.hour)
        .groupby("hour", as_index=False, observed=True)
        .agg(
            distance_mediane=("path_km", "median"),
            q25=("path_km", lambda series: series.quantile(0.25)),
            q75=("path_km", lambda series: series.quantile(0.75)),
            n_windows=("path_km", "size"),
        )
        .set_index("hour").reindex(range(24)).rename_axis("hour").reset_index()
    )
    return windows, hourly


def build_excursion_summary(data: pd.DataFrame, nest_radius_km: float, foraging_speed_kmh: float) -> pd.DataFrame:
    """Résume les séquences continues observées hors de la zone du nid."""

    movement = data.sort_values([cfg.ID_COLUMN, cfg.TIME_COLUMN]).copy()
    movement["outside_nest"] = movement["distance_nest_km"].gt(nest_radius_km)
    previous_outside = movement.groupby(cfg.ID_COLUMN, sort=False)["outside_nest"].shift(fill_value=False)
    movement["trip_start"] = movement["outside_nest"] & ~previous_outside
    movement["trip_id"] = movement.groupby(cfg.ID_COLUMN, sort=False)["trip_start"].cumsum()

    outside_points = movement.loc[movement["outside_nest"] & movement["trip_id"].gt(0)].copy()
    if outside_points.empty:
        return pd.DataFrame()

    trips = (
        outside_points.groupby([cfg.ID_COLUMN, "trip_id"], as_index=False, observed=True)
        .agg(
            start_time=(cfg.TIME_COLUMN, "min"),
            end_time=(cfg.TIME_COLUMN, "max"),
            n_positions=(cfg.TIME_COLUMN, "size"),
            maximum_range_km=("distance_nest_km", "max"),
            path_distance_km=("distance_km", "sum"),
            median_speed_kmh=("segment_speed_kmh", "median"),
            foraging_candidate_fraction=("ground_speed_kmh", lambda values: values.lt(foraging_speed_kmh).mean()),
        )
    )
    trips["duration_hours"] = (trips["end_time"] - trips["start_time"]).dt.total_seconds().div(3_600)
    trips["foraging_candidate_pct"] = 100 * trips["foraging_candidate_fraction"]
    trips["trip_label"] = "T" + trips["trip_id"].astype(int).astype(str)

    return trips.loc[trips["n_positions"].ge(cfg.MIN_TRIP_POSITIONS) & trips["duration_hours"].ge(cfg.MIN_TRIP_DURATION_HOURS)].reset_index(drop=True)


def add_turning_angles(data: pd.DataFrame) -> pd.DataFrame:
    """Calcule le cap des segments et l'angle entre deux segments successifs."""

    movement = data.sort_values([cfg.ID_COLUMN, cfg.TIME_COLUMN]).copy()
    grouped = movement.groupby(cfg.ID_COLUMN, sort=False)
    previous_lat = grouped[cfg.LAT_COLUMN].shift()
    previous_lon = grouped[cfg.LON_COLUMN].shift()

    lat_1 = np.radians(previous_lat)
    lat_2 = np.radians(movement[cfg.LAT_COLUMN])
    delta_lon = np.radians(movement[cfg.LON_COLUMN] - previous_lon)
    bearing_x = np.sin(delta_lon) * np.cos(lat_2)
    bearing_y = (np.cos(lat_1) * np.sin(lat_2) - np.sin(lat_1) * np.cos(lat_2) * np.cos(delta_lon))
    movement["bearing_deg"] = (np.degrees(np.arctan2(bearing_x, bearing_y)) + 360) % 360

    previous_bearing = movement.groupby(cfg.ID_COLUMN, sort=False)["bearing_deg"].shift()
    movement["turn_angle_deg"] = np.abs((movement["bearing_deg"] - previous_bearing + 180) % 360 - 180)
    return movement


def build_movement_signature_histogram(data: pd.DataFrame) -> Dict[str, object]:
    """Agrège vitesse et changement de direction dans une grille 2D."""

    movement = add_turning_angles(data)
    movement = movement.loc[movement["segment_speed_kmh"].ge(0) & movement["segment_speed_kmh"].notna() & movement["turn_angle_deg"].notna()].copy()

    if movement.empty:
        return {
            "counts": np.empty((0, 0)),
            "speed_centers": np.array([]),
            "turn_centers": np.array([]),
            "speed_cap": 20.0,
        }

    speed_cap = max(20.0, min(float(movement["segment_speed_kmh"].quantile(0.995)), 100.0))
    movement = movement.loc[movement["segment_speed_kmh"].le(speed_cap)]
    speed_edges = np.linspace(0, speed_cap, 35)
    turn_edges = np.linspace(0, 180, 31)
    counts, _, _ = np.histogram2d(movement["segment_speed_kmh"], movement["turn_angle_deg"], bins=[speed_edges, turn_edges])

    return {
        "counts": counts.T,
        "speed_centers": (speed_edges[:-1] + speed_edges[1:]) / 2,
        "turn_centers": (turn_edges[:-1] + turn_edges[1:]) / 2,
        "speed_cap": speed_cap,
    }


def estimate_typical_interval_minutes(filtered: pd.DataFrame) -> float:
    """Estime le pas GPS médian, borné entre une et quinze minutes."""

    typical_interval = float(filtered.loc[filtered["delta_time_hours"].gt(0), "delta_time_hours"].median() * 60)
    if not np.isfinite(typical_interval):
        return 3.0
    return float(np.clip(typical_interval, 1, 15))
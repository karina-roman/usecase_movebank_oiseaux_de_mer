# Détection, agrégation et synthèse des zones alimentaires potentielles

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg


def empty_foraging_stats() -> Dict[str, int]:
    """Retourne un dictionnaire de statistiques initialisé à zéro."""

    return {
        "candidate_points": 0,
        "clustered_points": 0,
        "noise_points": 0,
        "zones": 0,
    }


def _empty_zone_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "zone_nourriture",
            "center_lat",
            "center_lon",
            "n_points",
            "n_individuals",
            "median_speed_kmh",
            "first_timestamp",
            "last_timestamp",
        ]
    )


@st.cache_data(show_spinner="Détection des zones alimentaires potentielles…")
def detect_foraging_zones(data: pd.DataFrame, max_speed_kmh: float, nest_radius_km: float, dbscan_radius_m: int, min_samples: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Détecte les concentrations de positions lentes hors du nid avec DBSCAN."""

    try:
        from sklearn.cluster import DBSCAN
    except ImportError as error:
        raise RuntimeError("Le module scikit-learn est requis. Installez-le avec : python -m pip install scikit-learn") from error

    candidates = data.loc[data["ground_speed_kmh"].lt(max_speed_kmh) & data["distance_nest_km"].gt(nest_radius_km)].copy()
    candidates = candidates.reset_index(drop=True)
    candidates["au_nid"] = 0

    empty_summary = _empty_zone_summary()
    if len(candidates) < min_samples:
        candidates["zone_nourriture"] = -1
        stats = {
            "candidate_points": len(candidates),
            "clustered_points": 0,
            "noise_points": len(candidates),
            "zones": 0,
        }
        return candidates, empty_summary, stats

    coordinates = np.radians(candidates[[cfg.LAT_COLUMN, cfg.LON_COLUMN]].to_numpy())
    epsilon_radians = (dbscan_radius_m / 1_000) / cfg.EARTH_RADIUS_KM
    labels = DBSCAN(
        eps=epsilon_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit_predict(coordinates)

    # Z1 est la zone contenant le plus de positions, puis Z2, Z3, etc.
    cluster_counts = pd.Series(labels[labels >= 0]).value_counts()
    label_mapping = {original_label: rank for rank, original_label in enumerate(cluster_counts.index, start=1)}
    candidates["zone_nourriture"] = (pd.Series(labels, index=candidates.index).map(label_mapping).fillna(-1).astype(int))

    clustered = candidates.loc[candidates["zone_nourriture"].ne(-1)].copy()
    if clustered.empty:
        zone_summary = empty_summary
    else:
        zone_summary = (
            clustered.groupby(
                "zone_nourriture",
                as_index=False,
                observed=True,
            )
            .agg(
                center_lat=(cfg.LAT_COLUMN, "median"),
                center_lon=(cfg.LON_COLUMN, "median"),
                n_points=(cfg.ID_COLUMN, "size"),
                n_individuals=(cfg.ID_COLUMN, "nunique"),
                median_speed_kmh=("ground_speed_kmh", "median"),
                first_timestamp=(cfg.TIME_COLUMN, "min"),
                last_timestamp=(cfg.TIME_COLUMN, "max"),
            )
            .sort_values("n_points", ascending=False).reset_index(drop=True)
        )

    stats = {
        "candidate_points": len(candidates),
        "clustered_points": len(clustered),
        "noise_points": int(candidates["zone_nourriture"].eq(-1).sum()),
        "zones": len(zone_summary),
    }
    return candidates, zone_summary, stats


def select_displayed_zones(candidates: pd.DataFrame, zone_summary: pd.DataFrame, max_visible_zones: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sélectionne les zones les plus denses et leurs positions."""

    displayed_summary = zone_summary.head(max_visible_zones).copy()
    if displayed_summary.empty or candidates.empty:
        return displayed_summary, candidates.iloc[0:0].copy()

    displayed_ids = displayed_summary["zone_nourriture"].tolist()
    displayed_points = candidates.loc[candidates["zone_nourriture"].isin(displayed_ids)].copy()
    return displayed_summary, displayed_points


def build_heat_grid(map_data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Agrège les positions sur une grille spatiale pour la carte de chaleur."""

    heat_data = (
        map_data.assign(
            heat_lat=((map_data[cfg.LAT_COLUMN] / cfg.HEAT_GRID_DEGREES).round() * cfg.HEAT_GRID_DEGREES),
            heat_lon=((map_data[cfg.LON_COLUMN] / cfg.HEAT_GRID_DEGREES).round() * cfg.HEAT_GRID_DEGREES),
        )
        .groupby(["heat_lat", "heat_lon"], as_index=False, observed=True)
        .agg(position_count=(cfg.ID_COLUMN, "size"), individual_count=(cfg.ID_COLUMN, "nunique"),)
    )
    heat_data["heat_weight"] = np.log1p(heat_data["position_count"])
    reduction_pct = 100 * (1 - len(heat_data) / len(map_data))
    return heat_data, reduction_pct


def build_zone_visit_summary(zone_points: pd.DataFrame, typical_interval_minutes: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Regroupe les points consécutifs d'une zone en visites temporelles."""

    if zone_points.empty:
        return pd.DataFrame(), pd.DataFrame()

    visits_source = zone_points.loc[zone_points["zone_nourriture"].ne(-1)].sort_values(["zone_nourriture", cfg.ID_COLUMN, cfg.TIME_COLUMN]).copy()
    if visits_source.empty:
        return pd.DataFrame(), pd.DataFrame()

    time_gap_minutes = (visits_source.groupby(["zone_nourriture", cfg.ID_COLUMN], sort=False,)[cfg.TIME_COLUMN].diff().dt.total_seconds().div(60))
    visits_source["new_visit"] = (time_gap_minutes.isna() | time_gap_minutes.gt(cfg.VISIT_GAP_MINUTES))
    visits_source["visit_id"] = visits_source.groupby(["zone_nourriture", cfg.ID_COLUMN], sort=False,)["new_visit"].cumsum()

    visits = (
        visits_source.groupby(
            ["zone_nourriture", cfg.ID_COLUMN, "visit_id"],
            as_index=False,
            observed=True,
        )
        .agg(
            visit_start=(cfg.TIME_COLUMN, "min"),
            visit_end=(cfg.TIME_COLUMN, "max"),
            n_points=(cfg.TIME_COLUMN, "size"),
            median_speed_kmh=("ground_speed_kmh", "median"),
        )
    )
    visits["duration_minutes"] = (visits["visit_end"] - visits["visit_start"]).dt.total_seconds().div(60).add(typical_interval_minutes)
    visits["visit_midpoint"] = visits["visit_start"] + (visits["visit_end"] - visits["visit_start"]) / 2
    visits["zone_label"] = ("Z" + visits["zone_nourriture"].astype(int).astype(str))

    zone_use = (
        visits.groupby("zone_nourriture", as_index=False, observed=True)
        .agg(
            n_visits=("visit_id", "size"),
            n_individuals=(cfg.ID_COLUMN, "nunique"),
            n_active_days=("visit_start", lambda values: values.dt.normalize().nunique(),),
            total_time_hours=("duration_minutes", lambda values: values.sum() / 60,),
            median_visit_minutes=("duration_minutes", "median"),
            n_points=("n_points", "sum"),
        )
        .sort_values("n_points", ascending=False)
    )
    zone_use["zone_label"] = ("Z" + zone_use["zone_nourriture"].astype(int).astype(str))
    return visits, zone_use


def build_detection_histogram(filtered: pd.DataFrame, foraging_speed_kmh: float, nest_radius_km: float) -> Optional[Dict[str, object]]:
    """Agrège l'espace vitesse × éloignement utilisé avant DBSCAN."""

    detection_points = filtered.loc[filtered["ground_speed_kmh"].ge(0) & filtered["ground_speed_kmh"].notna() & filtered["distance_nest_km"].notna()].copy()
    if detection_points.empty:
        return None

    speed_cap = max(foraging_speed_kmh * 2, min(float(detection_points["ground_speed_kmh"].quantile(0.995)), 100.0))
    distance_cap = max(nest_radius_km * 2, float(detection_points["distance_nest_km"].quantile(0.995)))
    detection_plot = detection_points.loc[detection_points["ground_speed_kmh"].le(speed_cap) & detection_points["distance_nest_km"].le(distance_cap)]

    distance_edges = np.linspace(0, distance_cap, 40)
    speed_edges = np.linspace(0, speed_cap, 36)
    counts, _, _ = np.histogram2d(detection_plot["distance_nest_km"], detection_plot["ground_speed_kmh"], bins=[distance_edges, speed_edges])
    return {
        "counts": counts.T,
        "distance_centers": (distance_edges[:-1] + distance_edges[1:]) / 2,
        "speed_centers": (speed_edges[:-1] + speed_edges[1:]) / 2,
        "distance_cap": distance_cap,
        "speed_cap": speed_cap,
    }


def build_method_zone_table(displayed_summary: pd.DataFrame) -> pd.DataFrame:
    """Prépare le tableau descriptif présenté dans l'onglet méthodologique."""

    if displayed_summary.empty:
        return pd.DataFrame()

    table = displayed_summary.copy()
    table["zone"] = "Z" + table["zone_nourriture"].astype(int).astype(str)
    table["observation_days"] = (table["last_timestamp"] - table["first_timestamp"]).dt.total_seconds().div(86_400)
    return table[
        [
            "zone",
            "n_points",
            "n_individuals",
            "median_speed_kmh",
            "observation_days",
            "first_timestamp",
            "last_timestamp",
        ]
    ]
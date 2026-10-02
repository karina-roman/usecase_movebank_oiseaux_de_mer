# Fonctions de construction des cartes Plotly

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config as cfg
from visualization.charts  import improve_axis_spacing


def _map_center(map_data: pd.DataFrame) -> Dict[str, float]:
    return {
        "lat": float(map_data[cfg.LAT_COLUMN].median()),
        "lon": float(map_data[cfg.LON_COLUMN].median()),
    }


def _french_integer(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def build_map_config(mapbox_token: Optional[str]) -> Dict[str, object]:
    """Retourne la configuration commune aux cartes Plotly."""

    map_config = {
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
    }
    if mapbox_token:
        map_config["mapboxAccessToken"] = mapbox_token
    return map_config


def build_trajectory_map(map_data: pd.DataFrame, selected_map_style: str, show_markers: bool, color_map: Dict[str, str], individual_order: Iterable[str]) -> go.Figure:
    """Construit la carte normale avec trajectoires et positions GPS."""

    map_mode = "lines+markers" if show_markers else "lines"
    map_arguments = dict(
        data_frame=map_data,
        lat=cfg.LAT_COLUMN,
        lon=cfg.LON_COLUMN,
        color=cfg.ID_COLUMN,
        color_discrete_map=color_map,
        category_orders={cfg.ID_COLUMN: list(individual_order)},
        hover_name=cfg.ID_COLUMN,
        hover_data={
            cfg.ID_COLUMN: False,
            cfg.TIME_COLUMN: "|%d/%m/%Y %H:%M UTC",
            cfg.LAT_COLUMN: ":.4f",
            cfg.LON_COLUMN: ":.4f",
            "ground_speed_kmh": ":.1f",
            "segment_distance_km": ":.2f",
            "segment_speed_kmh": ":.1f",
            "distance_nest_km": ":.2f",
        },
        labels={
            cfg.ID_COLUMN: "Individu",
            cfg.TIME_COLUMN: "Horodatage",
            cfg.LAT_COLUMN: "Latitude",
            cfg.LON_COLUMN: "Longitude",
            "ground_speed_kmh": "Vitesse GPS (km/h)",
            "segment_distance_km": "Distance du segment (km)",
            "segment_speed_kmh": "Vitesse calculée (km/h)",
            "distance_nest_km": "Distance au nid estimé (km)",
        },
        center=_map_center(map_data),
        zoom=7.5,
        height=cfg.MAP_HEIGHT,
    )

    if hasattr(px, "scatter_map"):
        figure = px.scatter_map(**map_arguments, map_style=selected_map_style)
        revision_argument = {"map_uirevision": "keep-map-view"}
    else:
        figure = px.scatter_mapbox(**map_arguments, mapbox_style=selected_map_style)
        revision_argument = {"mapbox_uirevision": "keep-map-view"}

    figure.update_traces(
        mode=map_mode,
        line={"width": 2},
        marker={"size": 5, "opacity": 0.75},
    )
    figure.update_layout(
        title={
            "text": "Trajectoires GPS par individu",
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=False,
        margin={"r": 0, "t": 55, "l": 0, "b": 0},
        **revision_argument,
    )
    return improve_axis_spacing(figure)


def build_foraging_map(map_data: pd.DataFrame, heat_data: pd.DataFrame, displayed_zone_points: pd.DataFrame, displayed_zone_summary: pd.DataFrame, foraging_stats: Dict[str, int], 
                       selected_map_style: str, foraging_speed_kmh: float, nest_radius_km: float, dbscan_radius_m: int, dbscan_min_samples: int, heat_reduction_pct: float,) -> go.Figure:
    """Construit la chaleur spatiale et superpose les zones alimentaires."""

    heat_zmax = max(float(heat_data["heat_weight"].quantile(0.995)), 1.0)
    heat_customdata = heat_data[["position_count", "individual_count"]].to_numpy(dtype=object)
    heat_trace_arguments = dict(
        lat=heat_data["heat_lat"],
        lon=heat_data["heat_lon"],
        z=heat_data["heat_weight"],
        zmin=0,
        zmax=heat_zmax,
        radius=22,
        opacity=0.68,
        colorscale=[
            [0.00, "rgba(36, 44, 112, 0.00)"],
            [0.10, "rgba(53, 104, 218, 0.35)"],
            [0.32, "rgba(30, 203, 225, 0.58)"],
            [0.58, "rgba(250, 220, 65, 0.78)"],
            [0.80, "rgba(255, 112, 31, 0.90)"],
            [1.00, "rgba(181, 23, 23, 0.98)"],
        ],
        customdata=heat_customdata,
        hovertemplate=(
            "<b>Fréquentation spatiale</b>"
            "<br>Positions dans la maille : %{customdata[0]:.0f}"
            "<br>Individus : %{customdata[1]:.0f}"
            "<extra></extra>"
        ),
        colorbar={
            "title": {"text": "Fréquentation"},
            "len": 0.30,
            "thickness": 12,
            "x": 0.985,
            "y": 0.80,
            "outlinewidth": 0,
        },
        name="Fréquentation spatiale",
    )

    use_maplibre = hasattr(go, "Densitymap") and hasattr(go, "Scattermap")
    if use_maplibre:
        figure = go.Figure(go.Densitymap(**heat_trace_arguments))
        map_trace_class = go.Scattermap
        figure.update_layout(
            map={
                "style": selected_map_style,
                "center": _map_center(map_data),
                "zoom": 7,
                "uirevision": "keep-map-view",
            }
        )
    else:
        figure = go.Figure(go.Densitymapbox(**heat_trace_arguments))
        map_trace_class = go.Scattermapbox
        figure.update_layout(
            mapbox={
                "style": selected_map_style,
                "center": _map_center(map_data),
                "zoom": 7,
                "uirevision": "keep-map-view",
            }
        )

    figure.update_layout(
        title={
            "text": "Fréquentation spatiale des oiseaux",
            "x": 0.5,
            "xanchor": "center",
        },
        legend={
            "orientation": "v",
            "bgcolor": "rgba(255, 255, 255, 0.88)",
        },
        height=cfg.MAP_HEIGHT,
        margin={"r": 0, "t": 55, "l": 0, "b": 0},
    )

    if displayed_zone_points.empty:
        return improve_axis_spacing(figure)
    
    _add_foraging_zone_layers(
        figure=figure,
        map_trace_class=map_trace_class,
        map_data=map_data,
        heat_data=heat_data,
        displayed_zone_points=displayed_zone_points,
        displayed_zone_summary=displayed_zone_summary,
        foraging_stats=foraging_stats,
        foraging_speed_kmh=foraging_speed_kmh,
        nest_radius_km=nest_radius_km,
        dbscan_radius_m=dbscan_radius_m,
        dbscan_min_samples=dbscan_min_samples,
        heat_reduction_pct=heat_reduction_pct,
    )
    return improve_axis_spacing(figure)


def _add_foraging_zone_layers(figure: go.Figure, map_trace_class, map_data: pd.DataFrame, heat_data: pd.DataFrame, displayed_zone_points: pd.DataFrame, displayed_zone_summary: pd.DataFrame,
                              foraging_stats: Dict[str, int], foraging_speed_kmh: float, nest_radius_km: float, dbscan_radius_m: int, dbscan_min_samples: int, heat_reduction_pct: float) -> None:
    """Ajoute les points, centres de zones et nids à une carte de chaleur."""

    food_points = displayed_zone_points.copy()
    food_points["timestamp_label"] = food_points[cfg.TIME_COLUMN].dt.strftime("%d/%m/%Y %H:%M UTC")
    point_customdata = food_points[
        [
            cfg.ID_COLUMN,
            "timestamp_label",
            "ground_speed_kmh",
            "distance_nest_km",
            "zone_nourriture",
        ]
    ].to_numpy(dtype=object)
    figure.add_trace(
        map_trace_class(
            lat=food_points[cfg.LAT_COLUMN],
            lon=food_points[cfg.LON_COLUMN],
            mode="markers",
            marker={
                "size": 8,
                "color": cfg.FORAGING_ZONE_COLOR,
                "opacity": 0.80,
            },
            name="Lieux alimentaires potentiels",
            legendgroup="foraging-zones",
            showlegend=True,
            customdata=point_customdata,
            hovertemplate=(
                "<b>Lieu alimentaire potentiel - Z%{customdata[4]}</b>"
                "<br>Individu : %{customdata[0]}"
                "<br>Horodatage : %{customdata[1]}"
                "<br>Vitesse GPS : %{customdata[2]:.1f} km/h"
                "<br>Distance au nid : %{customdata[3]:.2f} km"
                "<extra></extra>"
            ),
        )
    )

    centroids = displayed_zone_summary.copy()
    centroids["first_label"] = centroids["first_timestamp"].dt.strftime("%d/%m/%Y %H:%M UTC")
    centroids["last_label"] = centroids["last_timestamp"].dt.strftime("%d/%m/%Y %H:%M UTC")
    centroid_customdata = centroids[
        [
            "zone_nourriture",
            "n_points",
            "n_individuals",
            "median_speed_kmh",
            "first_label",
            "last_label",
        ]
    ].to_numpy(dtype=object)
    centroid_sizes = np.clip(24 + 4 * np.log10(centroids["n_points"].clip(lower=1)), 28, 42)

    figure.add_trace(
        map_trace_class(
            lat=centroids["center_lat"],
            lon=centroids["center_lon"],
            mode="markers",
            marker={
                "size": (centroid_sizes + 10).tolist(),
                "color": "white",
                "opacity": 0.95,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        map_trace_class(
            lat=centroids["center_lat"],
            lon=centroids["center_lon"],
            mode="markers+text",
            marker={
                "size": centroid_sizes.tolist(),
                "color": cfg.FORAGING_ZONE_COLOR,
                "opacity": 0.98,
            },
            text=[f"Z{int(zone_id)}" for zone_id in centroids["zone_nourriture"]],
            textposition="middle center",
            textfont={"color": "white", "size": 11, "family": "Arial Black"},
            name=(f"Zones alimentaires potentielles ({len(displayed_zone_summary)} affichées)"),
            legendgroup="foraging-zones",
            showlegend=False,
            customdata=centroid_customdata,
            hovertemplate=(
                "<b>Zone Z%{customdata[0]}</b>"
                "<br>Positions : %{customdata[1]:.0f}"
                "<br>Individus : %{customdata[2]:.0f}"
                "<br>Vitesse médiane : %{customdata[3]:.1f} km/h"
                "<br>Première détection : %{customdata[4]}"
                "<br>Dernière détection : %{customdata[5]}"
                "<extra></extra>"
            ),
        )
    )

    nest_markers = (map_data.groupby(cfg.ID_COLUMN, as_index=False, observed=True).agg(nest_lat=("nest_lat", "first"), nest_lon=("nest_lon", "first")))
    figure.add_trace(
        map_trace_class(
            lat=nest_markers["nest_lat"],
            lon=nest_markers["nest_lon"],
            mode="markers",
            marker={"size": 17, "color": "white", "opacity": 0.95},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        map_trace_class(
            lat=nest_markers["nest_lat"],
            lon=nest_markers["nest_lon"],
            mode="markers",
            marker={"size": 10, "color": "#111827", "opacity": 1.0},
            name="Nids estimés",
            customdata=nest_markers[[cfg.ID_COLUMN]].to_numpy(dtype=object),
            hovertemplate=(
                "<b>Nid estimé</b><br>Individu : %{customdata[0]}"
                "<br>Méthode : première position valide<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
        "text": (
            f"Fréquentation spatiale et zones potentielles de recherche alimentaire<br><sup>Positions candidates : vitesse inférieure à {foraging_speed_kmh:g} km/h et distance au nid supérieure à "
            f"{nest_radius_km:g} km</sup><br><sup>DBSCAN utilise un rayon de {dbscan_radius_m} m et exige au moins {dbscan_min_samples} points dans ce voisinage</sup>"
        ),
            "x": 0.5,
            "xanchor": "center",
        },
        legend={
            "orientation": "h",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 10},
            "itemsizing": "constant",
            "bgcolor": "rgba(255, 255, 255, 0.88)",
        },
        margin={"r": 0, "t": 88, "l": 0, "b": 0},
    )
    figure.add_annotation(
        x=0.015,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            f"<b>{len(displayed_zone_summary)} zones principales affichées</b>"
            f"<br>{_french_integer(len(displayed_zone_points))} positions regroupées "
            f"sur {foraging_stats['zones']} zones détectées"
            f"<br>{_french_integer(len(map_data))} positions de passage → "
            f"{_french_integer(len(heat_data))} mailles "
            f"({heat_reduction_pct:.0f} % en moins)"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.94)",
        bordercolor=cfg.FORAGING_ZONE_COLOR,
        borderwidth=2,
        borderpad=6,
        font={"color": "#111827", "size": 11},
    )
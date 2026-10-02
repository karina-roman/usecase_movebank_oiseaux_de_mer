"""Fonctions de construction des figures Plotly autres que les cartes."""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config as cfg


def build_empty_figure(message: str, title: str, height: int,) -> go.Figure:
    """Construit une figure vide qui conserve la mise en page du dashboard."""

    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_layout(
        title={"text": title, "x": 0.5},
        height=height,
        margin={"r": 20, "t": 70, "l": 20, "b": 20},
    )
    return figure


def build_daily_distance_figure(daily_distance: pd.DataFrame, color_map: Dict[str, str], individual_order: Iterable[str],) -> go.Figure:
    """Construit la courbe de distance journalière."""

    figure = px.line(
        daily_distance,
        x="date",
        y="distance_km",
        color=cfg.ID_COLUMN,
        markers=True,
        color_discrete_map=color_map,
        category_orders={cfg.ID_COLUMN: list(individual_order)},
        labels={
            "date": "Date (UTC)",
            "distance_km": "Distance parcourue (km)",
            cfg.ID_COLUMN: "Individu",
        },
        height=cfg.TIME_CHART_HEIGHT,
    )
    figure.update_traces(line={"width": 2.5}, marker={"size": 7})
    figure.update_layout(
        title={
            "text": ("Distance parcourue chaque jour"),
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        hovermode="x unified",
        showlegend=False,
        margin={"r": 20, "t": 70, "l": 20, "b": 20},
        uirevision="keep-daily-view",
    )
    figure.update_xaxes(
        title=None,
        rangeslider_visible=True,
        rangeselector={
            "font": {"color": "black"},
            "buttons": [
                {
                    "count": 7,
                    "label": "7 j",
                    "step": "day",
                    "stepmode": "backward",
                },
                {
                    "count": 14,
                    "label": "14 j",
                    "step": "day",
                    "stepmode": "backward",
                },
                {
                    "step": "all",
                    "label": "Tout",
                },
            ],
        },
    )
    figure.update_yaxes(rangemode="tozero")
    return improve_axis_spacing(figure)


def build_hourly_cycle_figure(hourly: pd.DataFrame) -> go.Figure:
    """Construit la médiane horaire et son ruban interquartile Q25–Q75."""

    hourly = hourly
    customdata = np.column_stack([hourly["q25"], hourly["q75"], hourly["n_windows"]])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["q25"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            connectgaps=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["q75"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(83, 67, 54, 0.45)",
            name="Q25–Q75",
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                "Heure : %{x:.0f} h UTC"
                "<br>Q25 : %{customdata[0]:.2f} km"
                "<br>Q75 : %{customdata[1]:.2f} km"
                "<br>Fenêtres : %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
            connectgaps=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["distance_mediane"],
            mode="lines+markers",
            line={"color": "#91877F", "width": 2.5},
            marker={"color": "#615E5C", "size": 7},
            name="Distance médiane avec intervalle Q25–Q75",
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                "Heure : %{x:.0f} h UTC"
                "<br>Distance médiane : %{y:.2f} km"
                "<br>Fenêtres : %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
            connectgaps=False,
        )
    )
    figure.update_layout(
        title={
            "text": (f"Cycle horaire descriptif calculé avec des fenêtres de {cfg.WINDOW_MINUTES} minutes<br>"
                     "<sup>La courbe représente la distance médiane parcourue</sup><br><sup>La zone colorée représente l’intervalle entre Q25 et Q75</sup>"),
            "x": 0.5,
            "xanchor": "center",
        },
        height=cfg.TIME_CHART_HEIGHT,
        hovermode="x unified",
        showlegend=False,
        xaxis={
            "title": "Heure UTC",
            "tickmode": "array",
            "tickvals": list(range(0, 24, 2)),
            "range": [-0.5, 23.5],
        },
        yaxis={
            "title": "Distance par fenêtre (km)",
            "rangemode": "tozero",
        },
        margin={"r": 20, "t": 75, "l": 20, "b": 20},
        uirevision="keep-hourly-view",
    )
    return improve_axis_spacing(figure)


def build_excursion_figure(excursions: pd.DataFrame, color_map: Dict[str, str], individual_order: Iterable[str], nest_radius_km: float,) -> go.Figure:
    """Construit la typologie des excursions hors du nid."""

    if excursions.empty:
        return build_empty_figure(
            "Aucune excursion suffisamment documentée pour cette sélection.",
            "Typologie des excursions hors nid",
            cfg.BEHAVIOR_CHART_HEIGHT,
        )

    figure = px.scatter(
        excursions,
        x="duration_hours",
        y="maximum_range_km",
        color=cfg.ID_COLUMN,
        size="path_distance_km",
        color_discrete_map=color_map,
        category_orders={cfg.ID_COLUMN: list(individual_order)},
        size_max=30,
        hover_name=cfg.ID_COLUMN,
        hover_data={
            cfg.ID_COLUMN: False,
            "trip_label": True,
            "start_time": "|%d/%m/%Y %H:%M UTC",
            "end_time": "|%d/%m/%Y %H:%M UTC",
            "duration_hours": ":.2f",
            "maximum_range_km": ":.1f",
            "path_distance_km": ":.1f",
            "median_speed_kmh": ":.1f",
            "foraging_candidate_pct": ":.0f",
            "n_positions": True,
        },
        labels={
            cfg.ID_COLUMN: "Individu",
            "trip_label": "Excursion",
            "start_time": "Début",
            "end_time": "Fin",
            "duration_hours": "Durée hors nid (h)",
            "maximum_range_km": "Éloignement maximal (km)",
            "path_distance_km": "Distance de trajectoire (km)",
            "median_speed_kmh": "Vitesse médiane (km/h)",
            "foraging_candidate_pct": "Positions lentes (%)",
            "n_positions": "Positions",
        },

        height=cfg.BEHAVIOR_CHART_HEIGHT,
    )
    figure.update_traces(marker={"opacity": 0.76, "line": {"width": 0.7, "color": "white"}})
    figure.update_layout(
        title={
                "text": (f"Typologie des excursions hors du nid<br><sup>Excursion détectée au-delà de {nest_radius_km:g} km du nid estimé</sup><br><sup>"
                         "La taille du point représente la distance totale parcourue pendant l’excursion</sup><br><sup>Au total, {len(excursions)} excursions ont été analysées</sup>"),
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "pad": {"b": 20},
            },
        height=cfg.BEHAVIOR_CHART_HEIGHT,
        showlegend=False,
        margin={"r": 20, "t": 100, "l": 20, "b": 20},
        uirevision="keep-excursion-view",
    )
    figure.update_xaxes(rangemode="tozero")
    figure.update_yaxes(rangemode="tozero")
    return improve_axis_spacing(figure)


def build_movement_signature_figure(histogram: Dict[str, object], foraging_speed_kmh: float) -> go.Figure:
    """Construit la densité vitesse × changement de direction."""

    counts = np.asarray(histogram["counts"])
    if counts.size == 0:
        return build_empty_figure(
            "Pas assez de segments pour calculer la signature du mouvement.",
            "Signature du mouvement",
            cfg.BEHAVIOR_CHART_HEIGHT,
        )

    speed_cap = float(histogram["speed_cap"])
    figure = go.Figure(
        go.Heatmap(
            x=histogram["speed_centers"],
            y=histogram["turn_centers"],
            z=np.log1p(counts),
            customdata=counts.astype(int),
            colorscale="Viridis",
            colorbar={
                "title": {"text": "Densité"},
                "thickness": 12,
                "len": 0.72,
            },
            hovertemplate=(
                "Vitesse : %{x:.1f} km/h"
                "<br>Changement de direction : %{y:.0f}°"
                "<br>Segments dans la classe : %{customdata:.0f}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=foraging_speed_kmh,
        line={
            "color": cfg.FORAGING_ZONE_COLOR,
            "width": 2,
            "dash": "dash",
        },
        annotation_text=f"Seuil lent : {foraging_speed_kmh:g} km/h",
        annotation_position="top right",
        annotation_font_color="black",
    )
    figure.add_annotation(
        x=min(foraging_speed_kmh * 0.55, speed_cap * 0.12),
        y=154,
        text="Lent + tortueux<br>recherche locale possible",
        showarrow=False,
        bgcolor="rgba(190, 180, 185, 0.67)",
        font={"size": 10, "color": "black",},
    )
    figure.add_annotation(
        x=speed_cap * 0.76,
        y=18,
        text="Rapide + directionnel<br>transit possible",
        showarrow=False,
        bgcolor="rgba(101, 91, 96, 0.67)",
        font={"size": 10, "color": "black",},
    )
    figure.update_layout(
        title={
            "text": ("Signature du mouvement<br><sup>Croisement entre la vitesse et le changement de direction</sup><br><sup>Couleur = nombre de segments, affiché sur une échelle logarithmique</sup>"),
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=cfg.BEHAVIOR_CHART_HEIGHT,
        xaxis={"title": "Vitesse du segment (km/h)", "rangemode": "tozero"},
        yaxis={
            "title": "Changement absolu de direction (°)",
            "range": [0, 180],
        },
        margin={"r": 20, "t": 100, "l": 20, "b": 20},
        uirevision="keep-signature-view",
    )
    return improve_axis_spacing(figure)


def build_zone_reuse_figure(zone_use: pd.DataFrame) -> go.Figure:
    """Construit le graphique de fidélité et de partage des zones."""

    if zone_use.empty:
        return build_empty_figure(
            "Aucune zone disponible pour analyser la fidélité.",
            "Fidélité et partage des zones",
            cfg.ZONE_CHART_HEIGHT,
        )

    figure = px.scatter(
        zone_use,
        x="n_visits",
        y="n_individuals",
        size="total_time_hours",
        text="zone_label",
        size_max=42,
        color_discrete_sequence=[cfg.FORAGING_ZONE_COLOR],
        hover_name="zone_label",
        hover_data={
            "zone_label": False,
            "n_visits": True,
            "n_individuals": True,
            "n_active_days": True,
            "total_time_hours": ":.1f",
            "median_visit_minutes": ":.0f",
            "n_points": True,
        },
        labels={
            "n_visits": "Nombre de visites",
            "n_individuals": "Individus utilisateurs",
            "n_active_days": "Jours avec visite",
            "total_time_hours": "Temps cumulé estimé (h)",
            "median_visit_minutes": "Visite médiane (min)",
            "n_points": "Positions",
        },
        height=cfg.ZONE_CHART_HEIGHT,
    )
    figure.update_traces(
        textposition="top center",
        marker={
            "color": cfg.FORAGING_ZONE_COLOR,
            "opacity": 0.76,
            "line": {"width": 1, "color": "white"},
        },
    )
    figure.update_layout(
        title={
            "text": ("Fidélité et partage des zones<br><sup>Taille = temps cumulé estimé dans la zone</sup>"),
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=False,
        margin={"r": 20, "t": 78, "l": 20, "b": 20},
        uirevision="keep-zone-reuse-view",
    )
    figure.update_xaxes(rangemode="tozero", dtick=5)
    figure.update_yaxes(rangemode="tozero", dtick=1)
    return improve_axis_spacing(figure)


def build_zone_timeline_figure(zone_visits: pd.DataFrame, zone_use: pd.DataFrame, color_map: Dict[str, str], individual_order: Iterable[str]) -> go.Figure:
    """Construit la chronologie des visites des zones affichées."""

    if zone_visits.empty:
        return build_empty_figure(
            "Aucune visite disponible pour construire la chronologie.",
            "Chronologie des visites",
            cfg.ZONE_CHART_HEIGHT,
        )

    zone_order = zone_use.sort_values("zone_nourriture")["zone_label"].tolist()
    figure = px.scatter(
        zone_visits,
        x="visit_midpoint",
        y="zone_label",
        color=cfg.ID_COLUMN,
        size="duration_minutes",
        color_discrete_map=color_map,
        category_orders={cfg.ID_COLUMN: list(individual_order)},
        size_max=24,
        hover_name=cfg.ID_COLUMN,
        hover_data={
            cfg.ID_COLUMN: False,
            "zone_label": True,
            "visit_start": "|%d/%m/%Y %H:%M UTC",
            "visit_end": "|%d/%m/%Y %H:%M UTC",
            "duration_minutes": ":.0f",
            "n_points": True,
            "median_speed_kmh": ":.1f",
            "visit_midpoint": False,
        },
        labels={
            "zone_label": "Zone",
            "visit_start": "Début de visite",
            "visit_end": "Fin de visite",
            "duration_minutes": "Durée estimée (min)",
            "n_points": "Positions",
            "median_speed_kmh": "Vitesse médiane (km/h)",
        },
        height=cfg.ZONE_CHART_HEIGHT,
    )
    figure.update_layout(
        title={
            "text": (f"Chronologie des visites<br><sup>Nouvelle visite après plus de {cfg.VISIT_GAP_MINUTES} min sans position dans la zone</sup><br><sup>Taille du point = durée estimée de la visite</sup>"),
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=False,
        margin={"r": 20, "t": 78, "l": 20, "b": 20},
        uirevision="keep-zone-timeline-view",
    )
    figure.update_xaxes(title=None)
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(zone_order)),
    )
    return improve_axis_spacing(figure)


def build_detection_space_figure(histogram: Optional[Dict[str, object]], foraging_speed_kmh: float, nest_radius_km: float) -> Optional[go.Figure]:
    """Construit le graphique explicatif du filtre précédant DBSCAN."""

    if histogram is None:
        return None

    distance_cap = float(histogram["distance_cap"])
    figure = go.Figure(
        go.Heatmap(
            x=histogram["distance_centers"],
            y=histogram["speed_centers"],
            z=np.log1p(np.asarray(histogram["counts"])),
            customdata=np.asarray(histogram["counts"]).astype(int),
            colorscale="Cividis",
            colorbar={
                "title": {"text": "Densité"},
                "thickness": 12,
                "len": 0.72,
            },
            hovertemplate=(
                "Distance au nid : %{x:.1f} km"
                "<br>Vitesse GPS : %{y:.1f} km/h"
                "<br>Positions dans la classe : %{customdata:.0f}"
                "<extra></extra>"
            ),
        )
    )
    candidate_purple = "#7C3AED"

    figure.add_shape(
        type="rect",
        x0=nest_radius_km,
        x1=distance_cap,
        y0=0,
        y1=foraging_speed_kmh,
        line={
            "color": candidate_purple,
            "width": 2,
        },
        fillcolor="rgba(124, 58, 237, 0.18)",
        layer="above",
    )

    figure.add_annotation(
        x=min(nest_radius_km + distance_cap * 0.18, distance_cap * 0.55),
        y=foraging_speed_kmh * 0.50,
        text="Points candidats à DBSCAN",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.88)",
        font={
            "color": candidate_purple,
            "size": 11,
        },
    )
    figure.update_layout(
        title={
            "text": ("Sélection des positions candidates<br><sup>Le rectangle rose combine les deux conditions avant DBSCAN</sup>"),
            "x": 0.5,
            "xanchor": "center",
        },
        height=470,
        xaxis={"title": "Distance au nid estimé (km)"},
        yaxis={"title": "Vitesse GPS (km/h)"},
        margin={"r": 20, "t": 80, "l": 20, "b": 20},
        uirevision="keep-detection-space-view",
    )
    return improve_axis_spacing(figure)


def improve_axis_spacing(figure):
    """Évite la superposition des titres et des valeurs des axes."""

    figure.update_yaxes(
        automargin=True,
        title_standoff=12,
    )
    figure.update_xaxes(
        automargin=True,
        title_standoff=12,
    )

    # Réserve davantage d’espace à gauche et en bas.
    figure.update_layout(
        margin_l=90,
        margin_b=55,
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color="#81a8b6",
        ),
    )
    return figure
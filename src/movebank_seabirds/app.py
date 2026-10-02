import streamlit as st

# Cette commande doit rester avant les imports de modules utilisant Streamlit.
st.set_page_config(
    page_title="Trajectoires des oiseaux",
    page_icon="🪽",
    layout="wide",
)

import pandas as pd

# Importations constantes
import config as cfg

# Importations fonctions
from visualization.charts import (
    build_daily_distance_figure,
    build_detection_space_figure,
    build_excursion_figure,
    build_hourly_cycle_figure,
    build_movement_signature_figure,
    build_zone_reuse_figure,
    build_zone_timeline_figure,
)
from data.processing import (
    build_individual_summary,
    filter_trajectories,
    find_data_file,
    load_and_prepare_data,
    normalize_date_range,
)
from analysis.foraging import (
    build_detection_histogram,
    build_heat_grid,
    build_method_zone_table,
    build_zone_visit_summary,
    detect_foraging_zones,
    empty_foraging_stats,
    select_displayed_zones,
)
from visualization.maps import build_foraging_map, build_map_config, build_trajectory_map
from analysis.movement import (
    build_daily_distance,
    build_excursion_summary,
    build_hourly_summary,
    build_movement_signature_histogram,
    estimate_typical_interval_minutes,
)
from ui.components import (
    build_color_map,
    get_mapbox_token,
    render_dashboard,
    render_foraging_method,
    render_metrics,
    render_sidebar,
    render_summary_table,
)


def main() -> None:
    """Orchestre les filtres, calculs, figures et composants de l'application."""

    try:
        data_path = find_data_file()
        trajets, data_quality = load_and_prepare_data(str(data_path))
    except Exception as error:
        st.error(f"Impossible de charger les données : {error}")
        st.stop()
        return

    st.title("Trajectoires des oiseaux marins")
    st.caption("Exploration synchronisée des positions GPS et des distances journalières. Les distances sont calculées sur la sphère terrestre à partir des positions successives.")

    all_individuals = sorted(trajets[cfg.ID_COLUMN].dropna().unique().tolist())
    min_date = trajets[cfg.TIME_COLUMN].min().date()
    max_date = trajets[cfg.TIME_COLUMN].max().date()
    mapbox_token = get_mapbox_token()

    filters = render_sidebar(
        all_individuals=all_individuals,
        min_date=min_date,
        max_date=max_date,
        mapbox_token=mapbox_token,
    )

    selected_individuals = filters["selected_individuals"]
    if not selected_individuals:
        st.warning("Sélectionnez au moins un individu dans la barre latérale.")
        st.stop()
        return

    start_date, end_date = normalize_date_range(
        filters["selected_dates"],
        min_date,
        max_date,
    )
    filtered, _invalid_segment_count = filter_trajectories(
        trajets,
        selected_individuals,
        start_date,
        end_date,
    )
    if filtered.empty:
        st.warning("Aucune position ne correspond aux filtres sélectionnés.")
        st.stop()
        return

    individual_order = [
        identifier
        for identifier in all_individuals
        if identifier in selected_individuals
    ]

    color_map = build_color_map(all_individuals)
    map_data = filtered.sort_values([cfg.ID_COLUMN, cfg.TIME_COLUMN]).copy()
    map_config = build_map_config(mapbox_token)

    render_metrics(filtered, start_date, end_date)

    if filters["show_foraging_zones"]:
        render_foraging_mode(
            filtered=filtered,
            map_data=map_data,
            data_quality=data_quality,
            filters=filters,
            individual_order=individual_order,
            color_map=color_map,
            map_config=map_config,
        )

    else:
        render_standard_mode(
            filtered=filtered,
            map_data=map_data,
            filters=filters,
            individual_order=individual_order,
            color_map=color_map,
            map_config=map_config,
        )
    footer_html = (
    '<hr style="margin-top:3rem; margin-bottom:1rem;">'
    '<div style="text-align:center; color:#6b7280; font-size:0.82rem; line-height:1.7;">'

    'Dashboard créé dans le cadre du projet <a href="https://www.u-bordeaux.fr/universite/notre-strategie/'
    'nos-leviers/cma-competences-et-metiers-davenir/cap-ia" target="_blank"><strong>CAP IA</strong></a>.<br>'

    'Le code source est disponible sur <a href="https://github.com/cap-ia/usecase_movebank_oiseaux_de_mer" target="_blank">GitHub</a> '
    'sous <a href="https://opensource.org/license/mit" target="_blank">licence MIT</a>.<br>'

    'Données : Garthe et al. (2016), <a href="https://doi.org/10.5441/001/1.nk286sc0" target="_blank">'
    '« Terrestrial and Marine Foraging Strategies of an Opportunistic Seabird Species Breeding in the Wadden Sea »</a>, '
    'Movebank Data Repository, sous <a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank">licence CC0 1.0. Universal</a>.'

    '</div>'
)

    st.markdown(footer_html, unsafe_allow_html=True)



def render_standard_mode(*, filtered: pd.DataFrame, map_data: pd.DataFrame, filters: dict, individual_order: list, color_map: dict, map_config: dict,) -> None:
    """Calcule et affiche le mode normal avec trajectoires et analyses."""

    fig_map = build_trajectory_map(
        map_data=map_data,
        selected_map_style=filters["selected_map_style"],
        show_markers=filters["show_markers"],
        color_map=color_map,
        individual_order=individual_order,
    )

    daily_distance = build_daily_distance(filtered)
    _windows, hourly = build_hourly_summary(filtered)
    excursions = build_excursion_summary(
        filtered,
        nest_radius_km=filters["nest_radius_km"],
        foraging_speed_kmh=filters["foraging_speed_kmh"],
    )
    movement_histogram = build_movement_signature_histogram(filtered)

    fig_daily = build_daily_distance_figure(
        daily_distance,
        color_map,
        individual_order,
    )
    fig_hourly = build_hourly_cycle_figure(hourly)
    fig_excursions = build_excursion_figure(
        excursions,
        color_map,
        individual_order,
        nest_radius_km=filters["nest_radius_km"],
    )
    fig_movement_signature = build_movement_signature_figure(
        movement_histogram,
        foraging_speed_kmh=filters["foraging_speed_kmh"],
    )

    render_dashboard(
        show_foraging_zones=False,
        individual_order=individual_order,
        color_map=color_map,
        fig_map=fig_map,
        map_config=map_config,
        map_data=map_data,
        heat_data=pd.DataFrame(),
        heat_reduction_pct=0.0,
        foraging_error=None,
        displayed_zone_points=pd.DataFrame(),
        foraging_stats=empty_foraging_stats(),
        fig_daily=fig_daily,
        fig_hourly=fig_hourly,
        fig_excursions=fig_excursions,
        fig_movement_signature=fig_movement_signature,
    )

    summary = build_individual_summary(filtered, daily_distance)
    render_summary_table(summary)


def render_foraging_mode(*, filtered: pd.DataFrame, map_data: pd.DataFrame, data_quality: dict, filters: dict, individual_order: list, color_map: dict, map_config: dict,) -> None:
    """Calcule et affiche le mode chaleur, DBSCAN et visites de zones"""

    candidates = pd.DataFrame()
    zone_summary = pd.DataFrame()
    displayed_zone_summary = pd.DataFrame()
    displayed_zone_points = pd.DataFrame()
    foraging_stats = empty_foraging_stats()
    foraging_error = None

    zone_input_columns = [
        cfg.ID_COLUMN,
        cfg.TIME_COLUMN,
        cfg.LAT_COLUMN,
        cfg.LON_COLUMN,
        "ground_speed_kmh",
        "distance_nest_km",
        "nest_lat",
        "nest_lon",
    ]
    try:
        candidates, zone_summary, foraging_stats = detect_foraging_zones(
            filtered[zone_input_columns].copy(),
            max_speed_kmh=filters["foraging_speed_kmh"],
            nest_radius_km=filters["nest_radius_km"],
            dbscan_radius_m=filters["dbscan_radius_m"],
            min_samples=filters["dbscan_min_samples"],
        )
        displayed_zone_summary, displayed_zone_points = select_displayed_zones(
            candidates,
            zone_summary,
            max_visible_zones=filters["max_visible_zones"],
        )
    except Exception as error:
        foraging_error = str(error)

    heat_data, heat_reduction_pct = build_heat_grid(map_data)
    fig_map = build_foraging_map(
        map_data=map_data,
        heat_data=heat_data,
        displayed_zone_points=displayed_zone_points,
        displayed_zone_summary=displayed_zone_summary,
        foraging_stats=foraging_stats,
        selected_map_style=filters["selected_map_style"],
        foraging_speed_kmh=filters["foraging_speed_kmh"],
        nest_radius_km=filters["nest_radius_km"],
        dbscan_radius_m=filters["dbscan_radius_m"],
        dbscan_min_samples=filters["dbscan_min_samples"],
        heat_reduction_pct=heat_reduction_pct,
    )

    typical_interval_minutes = estimate_typical_interval_minutes(filtered)
    zone_visits, zone_use = build_zone_visit_summary(
        displayed_zone_points,
        typical_interval_minutes=typical_interval_minutes,
    )
    fig_zone_reuse = build_zone_reuse_figure(zone_use)
    fig_zone_timeline = build_zone_timeline_figure(
        zone_visits,
        zone_use,
        color_map,
        individual_order,
    )

    detection_histogram = build_detection_histogram(
        filtered,
        foraging_speed_kmh=filters["foraging_speed_kmh"],
        nest_radius_km=filters["nest_radius_km"],
    )
    fig_detection_space = build_detection_space_figure(
        detection_histogram,
        foraging_speed_kmh=filters["foraging_speed_kmh"],
        nest_radius_km=filters["nest_radius_km"],
    )
    method_zone_table = build_method_zone_table(displayed_zone_summary)

    analysis_tab, method_tab = st.tabs(["Résultats interactifs", "Méthode - zones alimentaires"])

    with analysis_tab:
        render_dashboard(
            show_foraging_zones=True,
            individual_order=individual_order,
            color_map=color_map,
            fig_map=fig_map,
            map_config=map_config,
            map_data=map_data,
            heat_data=heat_data,
            heat_reduction_pct=heat_reduction_pct,
            foraging_error=foraging_error,
            displayed_zone_points=displayed_zone_points,
            foraging_stats=foraging_stats,
            fig_zone_reuse=fig_zone_reuse,
            fig_zone_timeline=fig_zone_timeline,
        )

    with method_tab:
        render_foraging_method(
            foraging_stats=foraging_stats,
            data_quality=data_quality,
            foraging_speed_kmh=filters["foraging_speed_kmh"],
            nest_radius_km=filters["nest_radius_km"],
            dbscan_radius_m=filters["dbscan_radius_m"],
            dbscan_min_samples=filters["dbscan_min_samples"],
            max_visible_zones=filters["max_visible_zones"],
            typical_interval_minutes=typical_interval_minutes,
            method_zone_table=method_zone_table,
            fig_detection_space=fig_detection_space,
            foraging_error=foraging_error,
        )


if __name__ == "__main__":
    main()
    
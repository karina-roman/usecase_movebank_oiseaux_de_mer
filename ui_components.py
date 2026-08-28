# Composants Streamlit : filtres, mise en page, tableaux et méthode

from datetime import date
from html import escape
import os
from textwrap import dedent
from typing import Dict, Iterable, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg


def french_integer(value: float) -> str:
    """Formate un entier avec des espaces comme séparateurs de milliers."""

    return f"{int(round(value)):,}".replace(",", " ")


def get_mapbox_token() -> Optional[str]:
    """Lit le jeton Mapbox depuis Render ou .streamlit/secrets.toml."""

    environment_token = os.getenv("MAPBOX_TOKEN")
    if environment_token:
        return environment_token
    try:
        return st.secrets.get("MAPBOX_TOKEN")
    except Exception:
        return None


def build_color_map(individuals: Iterable[str]) -> Dict[str, str]:
    """Complète la palette si le fichier contient de nouveaux identifiants."""

    individuals = list(individuals)
    colors = dict(cfg.DEFAULT_COLORS)
    fallback_palette = px.colors.qualitative.Safe + px.colors.qualitative.Bold
    unknown_ids = [identifier for identifier in individuals if identifier not in colors]
    for index, identifier in enumerate(unknown_ids):
        colors[identifier] = fallback_palette[index % len(fallback_palette)]
    return colors


def render_shared_individual_legend(individuals: Iterable[str], colors: Dict[str, str]) -> None:
    """Affiche une seule légende compacte pour les visuels par individu."""

    legend_items = []
    for identifier in individuals:
        safe_identifier = escape(str(identifier))
        safe_color = escape(colors.get(identifier, "#6B7280"), quote=True)
        legend_items.append(
            "<span style='display:inline-flex; align-items:center; white-space:nowrap;'>"
            f"<span style='width:18px; height:4px; border-radius:4px;"
            f"background:{safe_color}; margin-right:5px;'></span>"
            f"{safe_identifier}</span>"
        )

    st.markdown(
        "<div style='display:flex;flex-wrap:wrap; align-items:center; gap:5px 13px; padding:6px 10px; margin:2px 0 8px 0;"
        "border:1px solid rgba(128,128,128,.25); border-radius:8px; font-size:.78rem; line-height:1.2;'>"
        "<span style='font-weight:600; margin-right:2px;'>Individus</span>" + "".join(legend_items) + "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(all_individuals: Iterable[str], min_date: date, max_date: date, mapbox_token: Optional[str]) -> Dict[str, object]:
    """Affiche tous les filtres et retourne leurs valeurs dans un dictionnaire."""

    all_individuals = list(all_individuals)
    available_map_styles = {
        "OpenStreetMap": "open-street-map",
        "Carto clair": "carto-positron",
        "Carto sombre": "carto-darkmatter",
    }
    if mapbox_token:
        available_map_styles["Satellite + routes"] = "satellite-streets"

    with st.sidebar:
        st.header("Filtres")
        selected_individuals = st.multiselect(
            "Individus",
            options=all_individuals,
            default=all_individuals,
        )
        selected_dates = st.date_input(
            "Période (UTC)",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        selected_map_label = st.selectbox(
            "Fond de carte",
            options=list(available_map_styles),
            index=0,
        )
        st.divider()

        show_foraging_zones = st.toggle(
            "Afficher les zones alimentaires potentielles",
            value=False,
            help=("Recherche les concentrations de positions lentes hors de la zone du nid à l'aide de DBSCAN."),
        )

        # Ce bouton apparaît uniquement en mode trajectoires.
        if not show_foraging_zones:
            show_markers = st.toggle("Afficher les points GPS", value=True)
        else:
            # La variable doit rester définie pour le dictionnaire retourné.
            show_markers = False

    
        foraging_speed_kmh = cfg.DEFAULT_FORAGING_SPEED_KMH
        nest_radius_km = cfg.DEFAULT_NEST_RADIUS_KM
        dbscan_radius_m = cfg.DEFAULT_DBSCAN_RADIUS_M
        dbscan_min_samples = cfg.DEFAULT_DBSCAN_MIN_SAMPLES
        max_visible_zones = cfg.DEFAULT_MAX_VISIBLE_ZONES

        if show_foraging_zones:
            with st.expander("Paramètres des zones", expanded=False):
                foraging_speed_kmh = st.slider(
                    "Vitesse maximale d'alimentation (km/h)",
                    min_value=0.5,
                    max_value=10.0,
                    value=cfg.DEFAULT_FORAGING_SPEED_KMH,
                    step=0.5,
                )
                nest_radius_km = st.slider(
                    "Rayon d'exclusion autour du nid (km)",
                    min_value=0.25,
                    max_value=5.0,
                    value=cfg.DEFAULT_NEST_RADIUS_KM,
                    step=0.25,
                )
                dbscan_radius_m = st.slider(
                    "Rayon DBSCAN (m)",
                    min_value=100,
                    max_value=2_000,
                    value=cfg.DEFAULT_DBSCAN_RADIUS_M,
                    step=100,
                )
                dbscan_min_samples = st.slider(
                    "Nombre minimal de points",
                    min_value=5,
                    max_value=100,
                    value=cfg.DEFAULT_DBSCAN_MIN_SAMPLES,
                    step=5,
                )
                max_visible_zones = st.slider(
                    "Nombre maximal de zones affichées",
                    min_value=5,
                    max_value=100,
                    value=cfg.DEFAULT_MAX_VISIBLE_ZONES,
                    step=5,
                    help="Les zones les plus riches en positions sont prioritaires.",
                )

            st.caption("Détection exploratoire : une zone indique une concentration de positions lentes, pas une preuve directe d'alimentation.")

        if not mapbox_token:
            st.caption("Pour activer le fond satellite, définissez `MAPBOX_TOKEN` dans `.streamlit/secrets.toml` en local ou comme variable d'environnement sur la plateforme de déploiement.")

    return {
        "selected_individuals": selected_individuals,
        "selected_dates": selected_dates,
        "selected_map_style": available_map_styles[selected_map_label],
        "show_markers": show_markers,
        "show_foraging_zones": show_foraging_zones,
        "foraging_speed_kmh": foraging_speed_kmh,
        "nest_radius_km": nest_radius_km,
        "dbscan_radius_m": dbscan_radius_m,
        "dbscan_min_samples": dbscan_min_samples,
        "max_visible_zones": max_visible_zones,
    }


def render_metrics(filtered: pd.DataFrame, start_date: date, end_date: date) -> None:
    """Affiche les quatre indicateurs situés au-dessus du dashboard."""

    tracking_days = (end_date - start_date).days + 1
    total_distance = filtered["distance_km"].sum()
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Individus", filtered[cfg.ID_COLUMN].nunique())
    metric_2.metric("Positions GPS", french_integer(len(filtered)))
    metric_3.metric("Distance cumulée", f"{french_integer(total_distance)} km")
    metric_4.metric("Période sélectionnée", f"{tracking_days} j")


def render_dashboard(*, show_foraging_zones: bool, individual_order: Iterable[str], color_map: Dict[str, str], fig_map: go.Figure, map_config: Dict[str, object], map_data: pd.DataFrame, heat_data: pd.DataFrame,
                     heat_reduction_pct: float, foraging_error: Optional[str], displayed_zone_points: pd.DataFrame, foraging_stats: Dict[str, int], fig_daily: Optional[go.Figure] = None, fig_hourly: Optional[go.Figure] = None,
                     fig_excursions: Optional[go.Figure] = None, fig_movement_signature: Optional[go.Figure] = None, fig_zone_reuse: Optional[go.Figure] = None, fig_zone_timeline: Optional[go.Figure] = None) -> None:
    """Affiche la carte et les figures correspondant au mode actif."""

    render_shared_individual_legend(individual_order, color_map)

    # Informations communes placées au-dessus des deux colonnes
    if show_foraging_zones:
        st.caption(
            f"Mode chaleur : {french_integer(len(map_data))} positions agrégées en {french_integer(len(heat_data))} mailles spatiales ({heat_reduction_pct:.0f} % de points cartographiques en moins).  \n"
            f"Zones alimentaires : {foraging_stats['candidate_points']} positions lentes hors nid, {foraging_stats['clustered_points']} positions regroupées dans {foraging_stats['zones']} zones."
        )

        if foraging_error:
            st.error(f"Détection des zones impossible : {foraging_error}")
        elif displayed_zone_points.empty:
            st.warning("Aucune zone ne satisfait les paramètres DBSCAN pour la sélection actuelle.")

    map_column, plots_column = st.columns([1.1, 1], gap="large")

    with map_column:
        st.plotly_chart(
            fig_map,
            use_container_width=True,
            theme=None,
            config=map_config,
        )

    with plots_column:
        if show_foraging_zones:
            st.plotly_chart(
                fig_zone_reuse,
                use_container_width=True,
                theme=None,
                config=cfg.PLOTLY_CONFIG,
            )
            st.plotly_chart(
                fig_zone_timeline,
                use_container_width=True,
                theme=None,
                config=cfg.PLOTLY_CONFIG,
            )
        else:
            st.plotly_chart(
                fig_daily,
                use_container_width=True,
                theme=None,
                config=cfg.PLOTLY_CONFIG,
            )
            st.plotly_chart(
                fig_hourly,
                use_container_width=True,
                theme=None,
                config=cfg.PLOTLY_CONFIG,
            )

    if not show_foraging_zones:
        st.subheader("Analyses comportementales complémentaires")
        st.caption("Les excursions décrivent l'échelle des sorties hors nid, la signature cinématique distingue les mouvements rapides et directionnels des mouvements lents et tortueux, sans leur attribuer automatiquement un comportement.")
        excursion_column, signature_column = st.columns(2, gap="large")
        with excursion_column:
            st.plotly_chart(
                fig_excursions,
                use_container_width=True,
                theme=None,
                config=cfg.PLOTLY_CONFIG,
            )
        with signature_column:
            st.plotly_chart(
                fig_movement_signature,
                use_container_width=True,
                theme=None,
                config=cfg.PLOTLY_CONFIG,
            )


def render_foraging_method(*, foraging_stats: Dict[str, int], data_quality: Dict[str, int], foraging_speed_kmh: float, nest_radius_km: float, dbscan_radius_m: int, dbscan_min_samples: int,
                           max_visible_zones: int, typical_interval_minutes: float, method_zone_table: pd.DataFrame, fig_detection_space: Optional[go.Figure], foraging_error: Optional[str]) -> None:
    """Affiche l'explication complète de la construction des clusters."""

    st.subheader("Comment les zones alimentaires potentielles sont-elles calculées ?")
    st.info("La détection identifie des lieux où les oiseaux restent lents et spatialement concentrés loin du nid. Elle mesure une recherche alimentaire potentielle, pas une capture de proie observée.")

    method_1, method_2, method_3, method_4 = st.columns(4)
    method_1.metric(
        "Positions candidates",
        french_integer(foraging_stats["candidate_points"]),
    )
    method_2.metric(
        "Positions regroupées",
        french_integer(foraging_stats["clustered_points"]),
    )
    method_3.metric("Clusters DBSCAN", foraging_stats["zones"])
    noise_share = (
        100 * foraging_stats["noise_points"] / foraging_stats["candidate_points"]
        if foraging_stats["candidate_points"]
        else 0
    )
    method_4.metric("Bruit DBSCAN", f"{noise_share:.1f} %")

    st.caption(
        f"Nettoyage initial : {french_integer(data_quality['raw_rows'])} lignes lues → {french_integer(data_quality['clean_rows'])} "
        f"positions valides, {french_integer(data_quality['invalid_positions'])} position incomplète ou hors limites et "
        f"{french_integer(data_quality['manual_outliers'])} anomalies signalées manuellement retirées."
    )
    st.markdown(
        "### Chaîne de calcul\n"
        "**Positions GPS nettoyées** → **distance au nid** → "
        "**filtre vitesse + éloignement** → **DBSCAN spatial** → "
        "**zones Z1, Z2… et bruit −1**"
    )

    with st.expander("1 - Sélection des positions candidates", expanded=True):
        st.markdown(
            dedent(
                f"""
                1. Les horodatages et coordonnées invalides sont retirés, ainsi que les positions marquées manuellement comme aberrantes.
                2. Pour chaque individu, le **nid est estimé par sa première position GPS valide**. C'est un proxy opérationnel, pas une localisation de nid vérifiée sur le terrain.
                3. La distance orthodromique entre chaque position et ce nid estimé est calculée avec la formule de Haversine.
                4. Une position devient candidate si sa vitesse GPS est **strictement inférieure à {foraging_speed_kmh:g} km/h** et si elle se trouve à **plus de {nest_radius_km:g} km du nid**.
                5. Seuls ces points candidats sont transmis à DBSCAN. Les autres positions restent visibles dans la couche de fréquentation, mais ne participent pas à la création des zones alimentaires.
                """
            ).strip()
        )

    epsilon_radians = (dbscan_radius_m / 1_000) / cfg.EARTH_RADIUS_KM
    with st.expander("2 - Construction et signification des clusters DBSCAN", expanded=True):
        st.markdown(
            dedent(
                f"""
                **DBSCAN regroupe les points selon leur densité spatiale sans imposer à l'avance un nombre de zones.**

                - Les coordonnées sont converties en radians et comparées avec la distance de **Haversine**, adaptée aux positions terrestres.
                - Le rayon de voisinage **`eps` vaut {dbscan_radius_m} m** (`{epsilon_radians:.8f}` radian).
                - Un **point cœur** possède au moins **{dbscan_min_samples} positions** (lui-même inclus) dans ce rayon.
                - Un **point frontière** ne possède pas forcément cette densité, mais il est assez proche d'un point cœur pour rejoindre sa zone.
                - Les points cœur reliés de proche en proche et leurs points frontières forment un même cluster.
                - Un point candidat trop isolé reçoit l'étiquette **`−1`** : il est considéré comme du bruit et n'est pas affiché comme zone.
                - Les clusters trouvés sont finalement renommés par abondance : **Z1** contient le plus de positions, puis **Z2**, **Z3**, etc. Ce numéro est un rang d'affichage, pas une mesure de qualité.
                """
            ).strip()
        )
        st.markdown(
            dedent(
                """
                | Paramètre | Valeur plus faible | Valeur plus élevée |
                |---|---|---|
                | Rayon `eps` | Zones plus petites ou fragmentées, davantage de bruit | Zones plus larges, avec un risque de fusion entre lieux voisins |
                | `min_samples` | Détection plus permissive, davantage de petits clusters | Détection plus stricte, moins de clusters et davantage de bruit |
                """
            ).strip()
        )

    if foraging_error:
        st.error(f"DBSCAN n'a pas pu être exécuté : {foraging_error}")
    elif fig_detection_space is not None:
        st.plotly_chart(
            fig_detection_space,
            use_container_width=True,
            theme=None,
            config=cfg.PLOTLY_CONFIG,
        )

    with st.expander("3 - Affichage des zones et construction des visites", expanded=True):
        st.markdown(
            dedent(
                f"""
                DBSCAN produit une étiquette numérique pour chaque position candidate. L'application transforme ensuite ces étiquettes en éléments lisibles sur la carte.

                - Les clusters sont classés selon leur nombre de positions : **Z1** est le plus abondant, puis Z2, Z3, etc.
                - Le centre affiché de chaque zone correspond à la latitude et à la longitude médianes de ses positions.
                - Pour préserver la lisibilité, seules les **{max_visible_zones} zones les plus abondantes** sont affichées, même si DBSCAN en détecte davantage.
                - Tous les points appartenant aux zones affichées utilisent la même couleur rose, leur zone exacte reste indiquée au survol.
                - La couche de chaleur est indépendante de DBSCAN : elle agrège **toutes les positions de passage** dans des mailles de `{cfg.HEAT_GRID_DEGREES}` degré pour alléger le navigateur.
                - Les candidats classés comme bruit **`−1`** ne sont pas affichés comme zone alimentaire.

                Pour construire la chronologie, les positions d'un même individu dans une même zone sont ensuite ordonnées dans le temps :
                - Une nouvelle visite commence après plus de **{cfg.VISIT_GAP_MINUTES} minutes** sans position dans cette zone
                - La durée estimée ajoute le pas GPS médian de **{typical_interval_minutes:.1f} min**, afin qu'une visite constituée d'un seul point ne soit pas considérée comme nulle
                - Le nombre de visites mesure la **fidélité à la zone**, tandis que le nombre d'individus indique son niveau de **partage**.
                """
            ).strip()
        )

    if not method_zone_table.empty:
        with st.expander("Résumé des clusters affichés sur la carte", expanded=False):
            st.dataframe(
                method_zone_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "zone": st.column_config.TextColumn("Zone"),
                    "n_points": st.column_config.NumberColumn("Positions", format="%d"),
                    "n_individuals": st.column_config.NumberColumn("Individus", format="%d"),
                    "median_speed_kmh": st.column_config.NumberColumn("Vitesse médiane", format="%.1f km/h"),
                    "observation_days": st.column_config.NumberColumn("Étendue temporelle", format="%.1f j"),
                    "first_timestamp": st.column_config.DatetimeColumn("Première détection", format="DD/MM/YYYY HH:mm"),
                    "last_timestamp": st.column_config.DatetimeColumn("Dernière détection", format="DD/MM/YYYY HH:mm"),
                },
            )

    with st.expander("4 - Interprétation et limites", expanded=True):
        st.markdown("Un cluster indique une **concentration de positions lentes loin du nid**. Il correspond donc à une zone de recherche alimentaire potentielle, et non à une capture de proie directement observée.")
        st.warning(
            "Le nid est estimé, les seuils influencent directement les résultats, DBSCAN ne tient ici compte que de l'espace et les positions successives sont autocorrélées. " \
            "Une validation par observation, accélérométrie, régime alimentaire ou données d'habitat serait nécessaire pour conclure à une alimentation effective."
        )

    with st.expander("Références méthodologiques", expanded=False):
        st.markdown(
            "- [Documentation officielle de DBSCAN - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) : définition de `eps`, `min_samples`, des clusters et du bruit.\n"
            "- [Bennison et al. (2018) - comparaison de méthodes de détection de la recherche alimentaire](https://doi.org/10.1002/ece3.3593) : une signature de recherche issue du mouvement reste un proxy du comportement.\n"
            "- [Schoombie et al. (2024) - influence de la fréquence d'échantillonnage GPS](https://doi.org/10.1186/s40462-024-00499-1) : les vitesses, angles et états inférés dépendent du pas temporel.\n"
            "- [Fauchald & Tveraa (2003) - recherche restreinte dans l'espace](https://doi.org/10.1890/0012-9658(2003)084%5B0282:UFPTIT%5D2.0.CO;2) : cadre écologique des concentrations locales de recherche."
        )


def render_summary_table(summary: pd.DataFrame) -> None:
    """Affiche la synthèse par individu dans un expander."""

    with st.expander("Synthèse par individu", expanded=False):
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                cfg.ID_COLUMN: st.column_config.TextColumn("Individu"),
                "positions": st.column_config.NumberColumn("Positions", format="%d"),
                "active_days": st.column_config.NumberColumn("Jours actifs", format="%d"),
                "total_distance_km": st.column_config.NumberColumn("Distance totale", format="%.1f km"),
                "mean_daily_distance_km": st.column_config.NumberColumn("Distance moyenne/jour", format="%.1f km"),
                "mean_speed_kmh": st.column_config.NumberColumn("Vitesse GPS moyenne", format="%.1f km/h"),
                "max_speed_kmh": st.column_config.NumberColumn("Vitesse GPS maximale", format="%.1f km/h"),
                "first_position": st.column_config.DatetimeColumn("Première position", format="DD/MM/YYYY HH:mm"),
                "last_position": st.column_config.DatetimeColumn("Dernière position", format="DD/MM/YYYY HH:mm"),
            },
        )
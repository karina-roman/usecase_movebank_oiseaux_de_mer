# Constantes partagées par l'application Streamlit

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
FILE_NAME = "FTZ_Foraging_in_lesser_black-backed_gulls.csv"

ID_COLUMN = "tag-local-identifier"
TIME_COLUMN = "timestamp"
LAT_COLUMN = "location-lat"
LON_COLUMN = "location-long"

EARTH_RADIUS_KM = 6_371.0088
WINDOW_MINUTES = 15
MIN_WINDOW_POSITIONS = 4
MIN_WINDOW_COVERAGE_MINUTES = 8

DEFAULT_FORAGING_SPEED_KMH = 5.0
DEFAULT_NEST_RADIUS_KM = 1.0
DEFAULT_DBSCAN_RADIUS_M = 500
DEFAULT_DBSCAN_MIN_SAMPLES = 20
DEFAULT_MAX_VISIBLE_ZONES = 20

HEAT_GRID_DEGREES = 0.003
FORAGING_ZONE_COLOR = "#615E5C"
VISIT_GAP_MINUTES = 30
MIN_TRIP_POSITIONS = 4
MIN_TRIP_DURATION_HOURS = 0.15

MAP_HEIGHT = 780
TIME_CHART_HEIGHT = 380
BEHAVIOR_CHART_HEIGHT = 470
ZONE_CHART_HEIGHT = 370

DEFAULT_COLORS = {
    "eo_mGPS2_041": "#FF4B8B",
    "eo_mGPS2_042": "#C53B38",
    "eo_mGPS2_043": "#14558E",
    "eo_mGPS2_044": "#43A047",
    "eo_mGPS2_045": "#FB8C00",
    "eo_mGPS2_046": "#8E24AA",
    "eo_mGPS2_047": "#00ACC1",
    "eo_mGPS2_048": "#652510",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}
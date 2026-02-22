from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Dashboard Kualitas Udara Beijing", page_icon="🌍", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "main_data.csv"

ZONE_ORDER = ["Perkotaan", "Pinggiran", "Perdesaan"]
ZONE_COLORS = {"Perkotaan": "#ef4444", "Pinggiran": "#f59e0b", "Perdesaan": "#22c55e"}
SEASON_ORDER = ["Musim Dingin", "Musim Semi", "Musim Panas", "Musim Gugur"]
WIND_DIR_FULL = {
    "N": "Utara",
    "NNE": "Utara-Timur Laut",
    "NE": "Timur Laut",
    "ENE": "Timur Laut-Timur",
    "E": "Timur",
    "ESE": "Timur-Tenggara",
    "SE": "Tenggara",
    "SSE": "Selatan-Tenggara",
    "S": "Selatan",
    "SSW": "Selatan-Barat Daya",
    "SW": "Barat Daya",
    "WSW": "Barat Daya-Barat",
    "W": "Barat",
    "WNW": "Barat Laut-Barat",
    "NW": "Barat Laut",
    "NNW": "Utara-Barat Laut",
}

AQI_LEVELS = [
    (0, 50, "Good", "#2E8B57", "Air quality is satisfactory, and air pollution poses little or no risk."),
    (51, 100, "Moderate", "#DAA520", "Air quality is acceptable; some pollutants may pose a minor risk for a small number of people."),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF8C00", "Sensitive groups may experience health effects; the general public is less likely to be affected."),
    (151, 200, "Unhealthy", "#DC143C", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."),
    (201, 300, "Very Unhealthy", "#8B008B", "Health alert: the risk of health effects is increased for everyone."),
    (301, 500, "Hazardous", "#800000", "Health warning of emergency conditions: everyone is more likely to be affected."),
]

AQI_CATEGORY_ORDER = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

AQI_COLOR_MAP = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}

# US EPA breakpoints untuk perhitungan sub-index AQI
PM25_AQI_BP = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
PM10_AQI_BP = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]
CO_AQI_BP = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]
NO2_AQI_BP = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]
SO2_AQI_BP = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]
O3_8H_AQI_BP = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]

STATION_COORDS = pd.DataFrame(
    {
        "station": [
            "Aotizhongxin",
            "Changping",
            "Dingling",
            "Dongsi",
            "Guanyuan",
            "Gucheng",
            "Huairou",
            "Nongzhanguan",
            "Shunyi",
            "Tiantan",
            "Wanliu",
            "Wanshouxigong",
        ],
        "lon": [116.397, 116.230, 116.220, 116.417, 116.339, 116.184, 116.628, 116.461, 116.655, 116.407, 116.287, 116.352],
        "lat": [39.982, 40.217, 40.292, 39.929, 39.929, 39.914, 40.328, 39.937, 40.127, 39.886, 39.987, 39.878],
        "zone": [
            "Perkotaan",
            "Pinggiran",
            "Perdesaan",
            "Perkotaan",
            "Perkotaan",
            "Pinggiran",
            "Perdesaan",
            "Perkotaan",
            "Pinggiran",
            "Perkotaan",
            "Perkotaan",
            "Perkotaan",
        ],
    }
)


def bulan_ke_musim(month: int) -> str:
    if month in [12, 1, 2]:
        return "Musim Dingin"
    if month in [3, 4, 5]:
        return "Musim Semi"
    if month in [6, 7, 8]:
        return "Musim Panas"
    return "Musim Gugur"


def ugm3_to_ppm(ugm3: pd.Series, mw: float) -> pd.Series:
    return (ugm3 * 24.45) / (mw * 1000.0)


def ugm3_to_ppb(ugm3: pd.Series, mw: float) -> pd.Series:
    return (ugm3 * 24.45) / mw


def aqi_subindex(series: pd.Series, breakpoints: list[tuple[float, float, int, int]]) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    for c_low, c_high, i_low, i_high in breakpoints:
        mask = (series >= c_low) & (series <= c_high)
        if mask.any():
            out.loc[mask] = ((i_high - i_low) / (c_high - c_low)) * (series.loc[mask] - c_low) + i_low
    out = out.clip(lower=0, upper=500)
    return out


def category_from_aqi(aqi: float) -> tuple[str, str, str]:
    if pd.isna(aqi):
        return "No Data", "#64748b", "No data available for this filter combination."
    for low, high, label, color, msg in AQI_LEVELS:
        if low <= aqi <= high:
            return label, color, msg
    return "Hazardous", "#800000", AQI_LEVELS[-1][4]


def full_wind_direction(wd_value) -> str:
    if pd.isna(wd_value):
        return "Tidak tersedia"
    wd_code = str(wd_value).strip().upper()
    return WIND_DIR_FULL.get(wd_code, str(wd_value))


def aqi_subindex_scalar(value: float, breakpoints: list[tuple[float, float, int, int]]) -> float:
    if pd.isna(value):
        return np.nan
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= value <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
    return np.nan


def severity_rank(label: str) -> int:
    rank_map = {
        "Good": 0,
        "Moderate": 1,
        "Unhealthy for Sensitive Groups": 2,
        "Unhealthy": 3,
        "Very Unhealthy": 4,
        "Hazardous": 5,
        "No Data": -1,
    }
    return rank_map.get(label, -1)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["hour_label"] = df["hour"].map(lambda h: f"{h:02d}:00")
    df["month_year"] = df["datetime"].dt.to_period("M").astype(str)
    df["month_date"] = pd.to_datetime(df["month_year"])
    df["season"] = df["month"].map(bulan_ke_musim)

    if "zone" not in df.columns:
        df = df.merge(STATION_COORDS[["station", "zone"]], on="station", how="left")

    # AQI resmi berbasis breakpoint dan durasi standar (tanpa ML)
    df = df.sort_values(["station", "datetime"]).copy()
    df["pm25_24h"] = df.groupby("station")["PM2.5"].transform(lambda s: s.rolling(24, min_periods=24).mean())
    df["pm10_24h"] = df.groupby("station")["PM10"].transform(lambda s: s.rolling(24, min_periods=24).mean())
    df["co_8h_ugm3"] = df.groupby("station")["CO"].transform(lambda s: s.rolling(8, min_periods=8).mean())
    df["o3_8h_ugm3"] = df.groupby("station")["O3"].transform(lambda s: s.rolling(8, min_periods=8).mean())

    df["co_8h_ppm"] = ugm3_to_ppm(df["co_8h_ugm3"], 28.01)
    df["o3_8h_ppm"] = ugm3_to_ppm(df["o3_8h_ugm3"], 48.0)
    df["no2_1h_ppb"] = ugm3_to_ppb(df["NO2"], 46.01)
    df["so2_1h_ppb"] = ugm3_to_ppb(df["SO2"], 64.07)

    df["aqi_pm25"] = aqi_subindex(df["pm25_24h"], PM25_AQI_BP)
    df["aqi_pm10"] = aqi_subindex(df["pm10_24h"], PM10_AQI_BP)
    df["aqi_co"] = aqi_subindex(df["co_8h_ppm"], CO_AQI_BP)
    df["aqi_no2"] = aqi_subindex(df["no2_1h_ppb"], NO2_AQI_BP)
    df["aqi_so2"] = aqi_subindex(df["so2_1h_ppb"], SO2_AQI_BP)
    df["aqi_o3"] = aqi_subindex(df["o3_8h_ppm"], O3_8H_AQI_BP)

    aqi_components = ["aqi_pm25", "aqi_pm10", "aqi_co", "aqi_no2", "aqi_so2", "aqi_o3"]
    df["aqi"] = df[aqi_components].max(axis=1, skipna=True)
    df["aqi_category"] = df["aqi"].apply(lambda x: category_from_aqi(x)[0])

    dominant_idx = df[aqi_components].idxmax(axis=1, skipna=True)
    dominant_map = {
        "aqi_pm25": "PM2.5",
        "aqi_pm10": "PM10",
        "aqi_co": "CO",
        "aqi_no2": "NO2",
        "aqi_so2": "SO2",
        "aqi_o3": "O3",
    }
    df["polutan_dominan"] = dominant_idx.map(dominant_map)
    df.loc[df["aqi"].isna(), "polutan_dominan"] = np.nan

    return df


def apply_filter(
    df: pd.DataFrame,
    stations: list[str],
    aqi_categories: list[str],
    hour_labels: list[str],
    seasons: list[str],
    start_date,
    end_date,
) -> pd.DataFrame:
    filtered = df.copy()

    if stations:
        filtered = filtered[filtered["station"].isin(stations)]
    if aqi_categories:
        filtered = filtered[filtered["aqi_category"].isin(aqi_categories)]
    if hour_labels:
        filtered = filtered[filtered["hour_label"].isin(hour_labels)]
    if seasons:
        filtered = filtered[filtered["season"].isin(seasons)]

    filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]
    return filtered


def build_map(station_summary: pd.DataFrame) -> folium.Map:
    m = folium.Map(
        location=[STATION_COORDS["lat"].mean(), STATION_COORDS["lon"].mean()],
        zoom_start=9,
        tiles="CartoDB positron",
    )

    valid = station_summary.dropna(subset=["PM2.5"]).copy()
    if valid.empty:
        return m

    q1, q2, q3 = valid["PM2.5"].quantile([0.25, 0.50, 0.75])
    map_zone_colors = {"Perkotaan": "#e74c3c", "Pinggiran": "#f1c40f", "Perdesaan": "#2ecc71"}

    def radius_by_quantile(val: float) -> int:
        if val <= q1:
            return 6
        if val <= q2:
            return 8
        if val <= q3:
            return 10
        return 12

    for _, row in valid.iterrows():
        color = map_zone_colors.get(row["zone"], "#555")
        tooltip = (
            f"{row['station']}<br>"
            f"Zone: {row['zone']}<br>"
            f"PM2.5 mean: {row['PM2.5']:.1f}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius_by_quantile(float(row["PM2.5"])),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            tooltip=tooltip,
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
         background: white; padding: 10px 12px; border: 1px solid #999;
         border-radius: 4px; font-size: 12px; color: #111 !important;">
    <strong style="color:#111 !important;">Zona</strong><br>
    <span style="display:inline-block; width:10px; height:10px; background:#e74c3c; margin-right:6px;"></span>Perkotaan<br>
    <span style="display:inline-block; width:10px; height:10px; background:#f1c40f; margin-right:6px;"></span>Pinggiran<br>
    <span style="display:inline-block; width:10px; height:10px; background:#2ecc71; margin-right:6px;"></span>Perdesaan
    </div>
    """

    size_legend = f"""
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 9999;
         background: white; padding: 10px 12px; border: 1px solid #999;
         border-radius: 4px; font-size: 12px; color: #111 !important;">
    <strong style="color:#111 !important;">Skala PM2.5 (kuantil)</strong><br>
    <span style="display:inline-block; width:12px; height:12px; border: 2px solid #333; border-radius: 50%;"></span>
    <span style="margin-left:6px;">≤ {q1:.1f}</span><br>
    <span style="display:inline-block; width:16px; height:16px; border: 2px solid #333; border-radius: 50%;"></span>
    <span style="margin-left:6px;">≤ {q2:.1f}</span><br>
    <span style="display:inline-block; width:20px; height:20px; border: 2px solid #333; border-radius: 50%;"></span>
    <span style="margin-left:6px;">≤ {q3:.1f}</span><br>
    <span style="display:inline-block; width:24px; height:24px; border: 2px solid #333; border-radius: 50%;"></span>
    <span style="margin-left:6px;">> {q3:.1f}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(size_legend))
    return m


def compute_aqi_multi_table(filtered: pd.DataFrame):
    base_df = filtered.sort_values(["station", "datetime"]).copy()

    labels6 = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    pm25_bins = [-np.inf, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
    pm10_bins = [-np.inf, 54.0, 154.0, 254.0, 354.0, 424.0, np.inf]
    co_bins = [-np.inf, 4.4, 9.4, 12.4, 15.4, 30.4, np.inf]
    no2_bins = [-np.inf, 53.0, 100.0, 360.0, 649.0, 1249.0, np.inf]
    so2_bins = [-np.inf, 35.0, 75.0, 185.0, 304.0, 604.0, np.inf]
    o3_8h_bins = [-np.inf, 0.054, 0.070, 0.085, 0.105, 0.200, np.inf]
    o3_8h_labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Outside 8h AQI scope",
    ]

    def ugm3_to_ppm(ugm3: pd.Series, mw: float) -> pd.Series:
        return (ugm3 * 24.45) / (mw * 1000.0)

    def ugm3_to_ppb(ugm3: pd.Series, mw: float) -> pd.Series:
        return (ugm3 * 24.45) / mw

    base_df["pm25_24h"] = base_df.groupby("station")["PM2.5"].transform(lambda s: s.rolling(24, min_periods=24).mean())
    base_df["pm10_24h"] = base_df.groupby("station")["PM10"].transform(lambda s: s.rolling(24, min_periods=24).mean())
    base_df["co_8h_ugm3"] = base_df.groupby("station")["CO"].transform(lambda s: s.rolling(8, min_periods=8).mean())
    base_df["o3_8h_ugm3"] = base_df.groupby("station")["O3"].transform(lambda s: s.rolling(8, min_periods=8).mean())

    base_df["co_8h_ppm"] = ugm3_to_ppm(base_df["co_8h_ugm3"], 28.01)
    base_df["o3_8h_ppm"] = ugm3_to_ppm(base_df["o3_8h_ugm3"], 48.0)
    base_df["no2_1h_ppb"] = ugm3_to_ppb(base_df["NO2"], 46.01)
    base_df["so2_1h_ppb"] = ugm3_to_ppb(base_df["SO2"], 64.07)

    base_df["pm25_cat"] = pd.cut(base_df["pm25_24h"], bins=pm25_bins, labels=labels6)
    base_df["pm10_cat"] = pd.cut(base_df["pm10_24h"], bins=pm10_bins, labels=labels6)
    base_df["co_cat"] = pd.cut(base_df["co_8h_ppm"], bins=co_bins, labels=labels6)
    base_df["no2_cat"] = pd.cut(base_df["no2_1h_ppb"], bins=no2_bins, labels=labels6)
    base_df["so2_cat"] = pd.cut(base_df["so2_1h_ppb"], bins=so2_bins, labels=labels6)
    base_df["o3_8h_cat"] = pd.cut(base_df["o3_8h_ppm"], bins=o3_8h_bins, labels=o3_8h_labels)

    def pct(series: pd.Series, labels: list[str]) -> pd.Series:
        return (series.value_counts(normalize=True).reindex(labels) * 100).fillna(0)

    summary_main = pd.DataFrame(
        {
            "PM2.5 (24h)": pct(base_df["pm25_cat"], labels6),
            "PM10 (24h)": pct(base_df["pm10_cat"], labels6),
            "CO (8h)": pct(base_df["co_cat"], labels6),
            "NO2 (1h)": pct(base_df["no2_cat"], labels6),
            "SO2 (1h)": pct(base_df["so2_cat"], labels6),
        }
    ).round(2)

    o3_table = pct(base_df["o3_8h_cat"], o3_8h_labels).to_frame("O3 (8h)").round(2)

    def severe_share(series: pd.Series, severe_labels: list[str]) -> float:
        valid = series.dropna()
        if valid.empty:
            return 0.0
        return float(valid.isin(severe_labels).mean() * 100)

    severe_main = {
        "PM2.5 (24h)": severe_share(base_df["pm25_cat"], ["Unhealthy", "Very Unhealthy", "Hazardous"]),
        "PM10 (24h)": severe_share(base_df["pm10_cat"], ["Unhealthy", "Very Unhealthy", "Hazardous"]),
        "CO (8h)": severe_share(base_df["co_cat"], ["Unhealthy", "Very Unhealthy", "Hazardous"]),
        "NO2 (1h)": severe_share(base_df["no2_cat"], ["Unhealthy", "Very Unhealthy", "Hazardous"]),
        "SO2 (1h)": severe_share(base_df["so2_cat"], ["Unhealthy", "Very Unhealthy", "Hazardous"]),
        "O3 (8h)": severe_share(base_df["o3_8h_cat"], ["Unhealthy", "Very Unhealthy"]),
    }

    return summary_main, o3_table, severe_main


st.markdown(
    """
    <style>
        .stApp { background-color: #050910; color: #f1f5f9; }
        section[data-testid="stSidebar"] { background-color: #0f172a; }
        h1, h2, h3, h4, h5, h6, p, span, label { color: #f8fafc !important; }
        .big-title { font-size: 3.0rem; font-weight: 900; margin-bottom: 0.1rem; letter-spacing: 0.3px; }
        .desc { color: #cbd5e1 !important; margin-bottom: 0.7rem; }
        div[data-testid="stMetricValue"] { color: #f8fafc; }
        div[data-testid="stMetricLabel"] > div { color: #cbd5e1; }
        .note-box {
            border: 1px solid #1e293b;
            background: #0b1220;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 10px;
            color: #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


df = load_data()

min_date = df["date"].min()
max_date = df["date"].max()
hour_options = [f"{h:02d}:00" for h in range(24)]
station_options = sorted(df["station"].dropna().unique().tolist())
category_options = AQI_CATEGORY_ORDER
season_options = [s for s in SEASON_ORDER if s in df["season"].dropna().unique().tolist()]

if "filter_state" not in st.session_state:
    st.session_state.filter_state = {
        "start_date": min_date,
        "end_date": max_date,
        "hour_choice": "Semua Jam",
        "season_choice": "Semua Musim",
        "stations": station_options,
        "category_choice": "Semua Kategori",
    }

with st.sidebar:
    st.markdown("<div style='text-align:center; font-size:64px;'>🧪</div>", unsafe_allow_html=True)
    with st.form("form_filter_dashboard"):
        st.markdown("### 📆 Rentang Tanggal")
        start_date_input = st.date_input(
            "Mulai",
            value=st.session_state.filter_state["start_date"],
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )
        end_date_input = st.date_input(
            "Selesai",
            value=st.session_state.filter_state["end_date"],
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )

        st.markdown("### 🕒 Pilih Jam")
        hour_choice_input = st.selectbox(
            "Jam",
            ["Semua Jam"] + hour_options,
            index=(["Semua Jam"] + hour_options).index(st.session_state.filter_state["hour_choice"])
            if st.session_state.filter_state["hour_choice"] in (["Semua Jam"] + hour_options)
            else 0,
            label_visibility="collapsed",
        )

        st.markdown("### 🍂 Pilih Musim")
        season_choice_input = st.selectbox(
            "Musim",
            ["Semua Musim"] + season_options,
            index=(["Semua Musim"] + season_options).index(st.session_state.filter_state["season_choice"])
            if st.session_state.filter_state["season_choice"] in (["Semua Musim"] + season_options)
            else 0,
            label_visibility="collapsed",
        )

        st.markdown("### 📍 Pilih Stasiun")
        stations_input = st.multiselect(
            "Stasiun",
            station_options,
            default=[s for s in st.session_state.filter_state["stations"] if s in station_options] or station_options,
            label_visibility="collapsed",
        )

        st.markdown("### 🏷️ Pilih Kategori AQI")
        category_choice_input = st.selectbox(
            "Kategori AQI",
            ["Semua Kategori"] + category_options,
            index=(["Semua Kategori"] + category_options).index(st.session_state.filter_state["category_choice"])
            if st.session_state.filter_state["category_choice"] in (["Semua Kategori"] + category_options)
            else 0,
            label_visibility="collapsed",
        )

        submit_filter = st.form_submit_button("▶️ Terapkan Filter", use_container_width=True)

    if submit_filter:
        st.session_state.filter_state = {
            "start_date": start_date_input,
            "end_date": end_date_input,
            "hour_choice": hour_choice_input,
            "season_choice": season_choice_input,
            "stations": stations_input if stations_input else station_options,
            "category_choice": category_choice_input,
        }
        st.session_state.filter_applied = True
        st.rerun()

start_date = st.session_state.filter_state["start_date"]
end_date = st.session_state.filter_state["end_date"]
hour_choice = st.session_state.filter_state["hour_choice"]
selected_hours = hour_options if hour_choice == "Semua Jam" else [hour_choice]
season_choice = st.session_state.filter_state["season_choice"]
selected_seasons = season_options if season_choice == "Semua Musim" else [season_choice]
selected_stations = st.session_state.filter_state["stations"]
category_choice = st.session_state.filter_state["category_choice"]
selected_categories = category_options if category_choice == "Semua Kategori" else [category_choice]

if start_date > end_date:
    st.error("Tanggal mulai tidak boleh lebih besar dari tanggal selesai.")
    st.stop()

filtered = apply_filter(df, selected_stations, selected_categories, selected_hours, selected_seasons, start_date, end_date)
if filtered.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini.")
    st.stop()

avg_aqi = float(filtered["aqi"].mean())
aqi_label, aqi_color, aqi_msg = category_from_aqi(avg_aqi)

st.markdown("<div class='big-title'>🌍 Dashboard Kualitas Udara Beijing</div>", unsafe_allow_html=True)

if st.session_state.get("filter_applied", False):
    st.success("Filter berhasil diterapkan.")
    st.session_state.filter_applied = False

# Ringkasan pemantauan inti
pm25_24h_mean = float(filtered["pm25_24h"].mean()) if "pm25_24h" in filtered.columns else np.nan
pm25_high_pct = float((filtered["PM2.5"] > 75).mean() * 100) if "PM2.5" in filtered.columns else np.nan
wind_dom = (
    full_wind_direction(filtered["wd"].mode().iloc[0])
    if ("wd" in filtered.columns and filtered["wd"].notna().any())
    else "Tidak tersedia"
)

pm25_aqi_est = aqi_subindex_scalar(pm25_24h_mean, PM25_AQI_BP)
pm25_aqi_label, pm25_aqi_color, _ = category_from_aqi(pm25_aqi_est)

overall_rank = severity_rank(aqi_label)
pm25_rank = severity_rank(pm25_aqi_label)
kontra_diagnosis = pm25_rank > overall_rank

if pd.isna(pm25_high_pct):
    jam_tinggi_text = "No Data"
else:
    jam_tinggi_text = f"{pm25_high_pct:.1f}%"

st.markdown("### ⚡ Insight")
st.caption(
    f"Periode: {pd.to_datetime(start_date).strftime('%Y/%m/%d')} - {pd.to_datetime(end_date).strftime('%Y/%m/%d')}"
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Rata-rata AQI", f"{avg_aqi:.1f}")
with k2:
    st.markdown(
        f"<div class='note-box'><b>Kategori AQI:</b> <span style='color:{aqi_color}; font-weight:700'>{aqi_label}</span></div>",
        unsafe_allow_html=True,
    )
with k3:
    st.metric("PM2.5 24h Rata-rata", f"{pm25_24h_mean:.1f}" if not pd.isna(pm25_24h_mean) else "No Data")
with k4:
    st.metric("Persentase Jam PM2.5 Tinggi", jam_tinggi_text)

insight_lines = [
    f"PM2.5 tinggi terjadi sekitar **{jam_tinggi_text}**, menunjukkan kondisi PM tinggi cukup sering.",
    f"Arah angin dominan saat periode ini adalah **{wind_dom}**.",
]
if kontra_diagnosis:
    insight_lines.insert(
        0,
        "Risiko **PM2.5** lebih tinggi dibandingkan indikator AQI rata-rata, jadi PM2.5 perlu jadi prioritas pengendalian.",
    )

st.markdown("**Poin penting:**")
for line in insight_lines:
    st.markdown(f"- {line}")

q1_tab, q2_tab, q3_tab, q4_tab = st.tabs(
    [
        "🧭 Area Prioritas",
        "⏰ Waktu Kritis",
        "🌦️ Dampak Cuaca",
        "🚨 Risiko AQI",
    ]
)

with q1_tab:
    st.markdown("### Area Prioritas")

    station_avg_pm25 = (
        filtered.groupby(["station", "zone"], as_index=False)["PM2.5"].mean().sort_values("PM2.5", ascending=False)
    )
    zone_avg_pm25 = filtered.groupby("zone", as_index=False)["PM2.5"].mean().sort_values("PM2.5", ascending=False)

    if (not station_avg_pm25.empty) and (not zone_avg_pm25.empty):
        top_station_name = station_avg_pm25.iloc[0]["station"]
        top_station_value = station_avg_pm25.iloc[0]["PM2.5"]
        top_zone_name = zone_avg_pm25.iloc[0]["zone"]
        top_zone_value = zone_avg_pm25.iloc[0]["PM2.5"]
        st.success(
            f"Insight utama: area prioritas saat ini adalah **{top_station_name}** "
            f"(rata-rata PM2.5 **{top_station_value:.2f}**), dan zona tertinggi adalah **{top_zone_name}** "
            f"(rata-rata **{top_zone_value:.2f}**)."
        )

    station_summary = station_avg_pm25.merge(STATION_COORDS[["station", "lat", "lon"]], on="station", how="left")
    st.markdown("#### Peta Area Prioritas")
    st_folium(build_map(station_summary), width=None, height=500)

    fig_station = px.bar(
        station_avg_pm25,
        x="station",
        y="PM2.5",
        color="zone",
        color_discrete_map=ZONE_COLORS,
        title="Ranking PM2.5 per Stasiun",
        labels={"station": "Stasiun", "PM2.5": "PM2.5", "zone": "Zona"},
        template="plotly_dark",
    )
    fig_station.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig_station, use_container_width=True)

    c1, c2 = st.columns([1.3, 1.0])
    with c1:
        st.markdown("**Top 5 Stasiun Prioritas (PM2.5 tertinggi)**")
        st.dataframe(station_avg_pm25.head(5), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Ringkasan PM2.5 per Zona**")
        st.dataframe(zone_avg_pm25, use_container_width=True, hide_index=True)

with q2_tab:
    st.markdown("### Waktu Kritis")
    polutan_opsi = ["NO2", "CO", "SO2", "O3"]
    polutan_terpilih = st.selectbox("Pilih Polutan", polutan_opsi, index=0)

    monthly_pol = filtered.groupby("month_date", as_index=False)[polutan_terpilih].mean().sort_values("month_date")
    fig_month = px.line(
        monthly_pol,
        x="month_date",
        y=polutan_terpilih,
        markers=True,
        title=f"Tren Bulanan {polutan_terpilih}",
        labels={"month_date": "Bulan", polutan_terpilih: "Konsentrasi"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_month, use_container_width=True)

    pivot = filtered.pivot_table(index="hour", columns="month", values=polutan_terpilih, aggfunc="mean")
    fig_heat = go.Figure(
        data=[
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="YlOrRd",
                colorbar=dict(title="Konsentrasi"),
            )
        ]
    )
    fig_heat.update_layout(
        title=f"Heatmap Jam vs Bulan untuk {polutan_terpilih}",
        template="plotly_dark",
        xaxis_title="Bulan",
        yaxis_title="Jam",
        height=500,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    season_single = filtered[["season", polutan_terpilih]].dropna().copy()
    season_single = season_single.rename(columns={polutan_terpilih: "Konsentrasi"})
    fig_season = px.box(
        season_single,
        x="season",
        y="Konsentrasi",
        color="season",
        category_orders={"season": SEASON_ORDER},
        title=f"Distribusi Musiman {polutan_terpilih}",
        template="plotly_dark",
    )
    fig_season.update_xaxes(title="Musim")
    st.plotly_chart(fig_season, use_container_width=True)

    month_peak_value = int(filtered.groupby("month")[polutan_terpilih].mean().idxmax())
    hour_peak_value = int(filtered.groupby("hour")[polutan_terpilih].mean().idxmax())
    peak_selected = pd.DataFrame(
        [
            {
                "polutan": polutan_terpilih,
                "peak_month": month_peak_value,
                "peak_hour": f"{hour_peak_value:02d}:00",
            }
        ]
    )
    st.success(
        f"Insight utama: untuk **{polutan_terpilih}**, puncak rata-rata terjadi pada "
        f"**bulan {month_peak_value}** dan sekitar **jam {hour_peak_value:02d}:00**."
    )
    with st.expander("🔎 Lihat tabel ringkasan puncak"):
        st.dataframe(peak_selected, use_container_width=True, hide_index=True)

with q3_tab:
    st.markdown("### Dampak Cuaca")

    assoc_df = filtered.copy()
    assoc_df["season_en"] = assoc_df["month"].map(
        {
            12: "WINTER", 1: "WINTER", 2: "WINTER",
            3: "SPRING", 4: "SPRING", 5: "SPRING",
            6: "SUMMER", 7: "SUMMER", 8: "SUMMER",
            9: "AUTUMN", 10: "AUTUMN", 11: "AUTUMN",
        }
    )
    season_order = ["SPRING", "SUMMER", "AUTUMN", "WINTER"]
    pm_cols = ["PM2.5", "PM10"]
    gas_cols = ["CO", "NO2", "SO2", "O3"]
    met_cols = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

    def corr_by_group(data: pd.DataFrame, group_col: str, left_cols: list[str], right_cols: list[str]) -> pd.DataFrame:
        rows = []
        for g in season_order if group_col == "season_en" else sorted(data[group_col].dropna().unique().tolist()):
            sub = data[data[group_col] == g]
            if sub.empty:
                continue
            corr = sub[left_cols + right_cols].corr(method="spearman")
            for l in left_cols:
                for r in right_cols:
                    rows.append({"group": g, "left": l, "right": r, "rho": corr.loc[l, r]})
        return pd.DataFrame(rows)

    season_pm_met = corr_by_group(assoc_df, "season_en", pm_cols, met_cols)
    season_gas_met = corr_by_group(assoc_df, "season_en", gas_cols, met_cols)

    # Ringkasan cepat untuk end-user
    if (not season_pm_met.empty) and (not season_gas_met.empty):
        pm_peak = season_pm_met.loc[season_pm_met["rho"].abs().idxmax()]
        gas_peak = season_gas_met.loc[season_gas_met["rho"].abs().idxmax()]
        arah_pm = "naik" if pm_peak["rho"] > 0 else "turun"
        arah_gas = "naik" if gas_peak["rho"] > 0 else "turun"
        st.info(
            f"Insight cepat: {pm_peak['left']} cenderung {arah_pm} saat {pm_peak['right']} berubah "
            f"(musim {pm_peak['group']}, nilai asosiasi {pm_peak['rho']:.2f}). "
            f"Untuk gas, pola paling kuat ada pada {gas_peak['left']} vs {gas_peak['right']} "
            f"(musim {gas_peak['group']}, nilai {gas_peak['rho']:.2f})."
        )

    pm_focus = st.selectbox("Fokus PM", ["PM2.5", "PM10"], index=0)
    pm_focus_met = (
        season_pm_met[season_pm_met["left"] == pm_focus]
        .pivot(index="right", columns="group", values="rho")
        .reindex(columns=season_order)
    )

    cgas1, cgas2 = st.columns(2)
    with cgas1:
        gas_focus = st.selectbox("Filter Gas", ["Semua Gas"] + gas_cols, index=0)
    with cgas2:
        met_focus = st.selectbox("Filter Faktor Cuaca", ["Semua Faktor"] + met_cols, index=0)

    gas_met = season_gas_met.copy()
    if gas_focus != "Semua Gas":
        gas_met = gas_met[gas_met["left"] == gas_focus]
    if met_focus != "Semua Faktor":
        gas_met = gas_met[gas_met["right"] == met_focus]
    gas_met["pair"] = gas_met["left"] + " vs " + gas_met["right"]
    gas_met_pivot = gas_met.pivot(index="pair", columns="group", values="rho").reindex(columns=season_order)

    fig_pm_met = px.imshow(
        pm_focus_met,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"{pm_focus} vs Meteorologi per Musim",
    )
    fig_pm_met.update_layout(template="plotly_dark")
    st.plotly_chart(fig_pm_met, use_container_width=True)

    if gas_met_pivot.empty:
        st.warning("Tidak ada data untuk kombinasi filter gas dan faktor cuaca yang dipilih.")
    else:
        title_gas_met = "Gas vs Meteorologi per Musim"
        if gas_focus != "Semua Gas" or met_focus != "Semua Faktor":
            title_gas_met = f"{gas_focus} vs {met_focus} per Musim" if (gas_focus != "Semua Gas" and met_focus != "Semua Faktor") else (
                f"{gas_focus} vs Meteorologi per Musim" if gas_focus != "Semua Gas" else f"Gas vs {met_focus} per Musim"
            )
        fig_gas_met = px.imshow(
            gas_met_pivot,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title=title_gas_met,
        )
        fig_gas_met.update_layout(template="plotly_dark", height=400 if len(gas_met_pivot) <= 4 else 700)
        st.plotly_chart(fig_gas_met, use_container_width=True)

    wd_summary = assoc_df.groupby(["season_en", "wd"], as_index=False)[["PM2.5", "O3"]].mean()
    wd_summary["arah_angin"] = wd_summary["wd"].map(full_wind_direction)
    top_wd_pm25 = wd_summary.sort_values(["season_en", "PM2.5"], ascending=[True, False]).groupby("season_en").head(3)
    fig_wd = px.bar(
        top_wd_pm25,
        x="arah_angin",
        y="PM2.5",
        facet_col="season_en",
        facet_col_wrap=2,
        color="season_en",
        title="Top Arah Angin dengan PM2.5 Tertinggi per Musim",
        template="plotly_dark",
    )
    st.plotly_chart(fig_wd, use_container_width=True)

with q4_tab:
    st.markdown("### Risiko AQI")
    st.caption("Kategori AQI mengikuti standar resmi US EPA.")

    _, _, severe_main = compute_aqi_multi_table(filtered)
    if severe_main:
        top_risk_name, top_risk_value = max(severe_main.items(), key=lambda x: x[1])
        st.success(
            f"Insight utama: polutan dengan proporsi kategori berisiko tertinggi adalah "
            f"**{top_risk_name}** (sekitar **{top_risk_value:.2f}%** observasi)."
        )

    # Visual utama: pie overall + ranking risiko polutan
    category_counts = filtered["aqi_category"].value_counts().reindex(AQI_CATEGORY_ORDER, fill_value=0)
    category_counts = category_counts[category_counts > 0]
    st.markdown("### Distribusi Kategori AQI (Overall)")
    if category_counts.empty:
        st.warning("Distribusi kategori AQI tidak tersedia untuk filter saat ini.")
    else:
        fig_dist = go.Figure(
            data=[
                go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hole=0.35,
                    marker=dict(
                        colors=[AQI_COLOR_MAP[c] for c in category_counts.index],
                        line=dict(color="white", width=1),
                    ),
                    textinfo="label+percent",
                )
            ]
        )
        fig_dist.update_layout(
            title="Distribusi Kategori AQI Overall",
            template="plotly_dark",
            height=480,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    severe_df = pd.DataFrame({"polutan": list(severe_main.keys()), "severe_pct": list(severe_main.values())}).sort_values(
        "severe_pct", ascending=False
    )
    fig_severe = px.bar(
        severe_df,
        x="polutan",
        y="severe_pct",
        title="Peringkat Polutan pada Level Berisiko (%)",
        labels={"polutan": "Polutan", "severe_pct": "Unhealthy+ (%)"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_severe, use_container_width=True)

    if not severe_df.empty:
        top = severe_df.iloc[0]
        st.success(f"Polutan paling sering berada pada level berisiko: {top['polutan']} ({top['severe_pct']:.2f}%).")

    st.markdown("### 3 Stasiun Terbaik vs Terburuk (AQI)")
    district_aqi = filtered.groupby("station")["aqi"].mean().sort_values()
    col_best, col_worst = st.columns(2)
    with col_best:
        st.success("🌟 3 Stasiun Terbaik (AQI terendah)")
        for i, (station, val) in enumerate(district_aqi.head(3).items(), start=1):
            st.write(f"{i}. **{station}**: {val:.2f}")
    with col_worst:
        st.error("⚠️ 3 Stasiun Terburuk (AQI tertinggi)")
        for i, (station, val) in enumerate(district_aqi.tail(3).iloc[::-1].items(), start=1):
            st.write(f"{i}. **{station}**: {val:.2f}")

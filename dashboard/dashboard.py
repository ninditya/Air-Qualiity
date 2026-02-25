from pathlib import Path
import base64
import io
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_READY = True
except Exception:
    folium = None
    st_folium = None
    FOLIUM_READY = False

st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌍",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent / "main_data.csv"
CACHE_VERSION = "2026-02-25-co-aqi-fix-v1"

HISTORY_METRICS = ["AQI", "PM2.5", "PM10", "O3", "CO", "NO2", "SO2"]
POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
AQI_COLOR_MAP = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
    "No Data": "#9E9E9E",
}
AQI_COLOR_MAP_PASTEL_TEGAS = {
    "Good": "#7FBF9A",
    "Moderate": "#D9C87A",
    "Unhealthy for Sensitive Groups": "#D6A46F",
    "Unhealthy": "#C88686",
    "Very Unhealthy": "#A98BB7",
    "Hazardous": "#8D6A77",
    "No Data": "#A7ADB4",
}
AQI_HEALTH_IMPACT = {
    "Good": "Kualitas udara baik, aman untuk aktivitas luar ruang.",
    "Moderate": "Kelompok sensitif sebaiknya mengurangi aktivitas berat di luar.",
    "Unhealthy for Sensitive Groups": "Kelompok sensitif berisiko, batasi paparan luar ruang.",
    "Unhealthy": "",
    "Very Unhealthy": "Risiko kesehatan tinggi, hindari aktivitas luar ruang.",
    "Hazardous": "Darurat kesehatan, tetap di dalam ruangan jika memungkinkan.",
    "No Data": "Data belum cukup untuk menilai dampak kesehatan.",
}
STATION_COORDS = {
    "Aotizhongxin": (39.982, 116.406),
    "Changping": (40.218, 116.231),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.939, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}
STATION_ZONE_MAP = {
    "Aotizhongxin": "Urban",
    "Dongsi": "Urban",
    "Guanyuan": "Urban",
    "Nongzhanguan": "Urban",
    "Tiantan": "Urban",
    "Wanshouxigong": "Urban",
    "Changping": "Suburban",
    "Gucheng": "Suburban",
    "Shunyi": "Suburban",
    "Wanliu": "Suburban",
    "Dingling": "Rural",
    "Huairou": "Rural",
}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SUBINDEX_MAP = {
    "PM2.5": "aqi_pm25",
    "PM10": "aqi_pm10",
    "NO2": "aqi_no2",
    "SO2": "aqi_so2",
    "CO": "aqi_co",
    "O3": "aqi_o3",
}
ID_MONTH = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}

# Soft palette for charts
SOFT_PRIMARY = "#4E7394"
SOFT_NEUTRAL_DARK = "#4A5C6E"
SOFT_HEAT_COLORSCALE = [[0.0, "#94B8A4"], [0.5, "#D6BE84"], [1.0, "#B06F6F"]]
SOFT_DIVERGING_COLORSCALE = [[0.0, "#6D8FB3"], [0.5, "#F4F6F8"], [1.0, "#B47A8A"]]
SOFT_SEASON_COLORS = {
    "Winter": "#6F8FAF",
    "Spring": "#6EAA8E",
    "Summer": "#C39A62",
    "Autumn": "#B57A67",
}
SOFT_ZONE_COLORS = {"Urban": "#A76565", "Suburban": "#B8924F", "Rural": "#5F8D72"}

# 2) AQI CALCULATION
def _aqi_subindex_series(concentration: pd.Series, breakpoints: list[tuple[float, float, int, int]]) -> pd.Series:
    out = pd.Series(np.nan, index=concentration.index, dtype="float64")
    for c_low, c_high, i_low, i_high in breakpoints:
        mask = (concentration >= c_low) & (concentration <= c_high)
        if mask.any():
            out.loc[mask] = ((i_high - i_low) / (c_high - c_low)) * (concentration.loc[mask] - c_low) + i_low
    return out.clip(0, 500)


def _co_to_ppm_if_needed(co_series: pd.Series) -> pd.Series:
    """Convert CO to ppm when the input appears to be in µg/m³."""
    c = pd.to_numeric(co_series, errors="coerce").copy()
    if c.dropna().empty:
        return c
    # If values exceed AQI CO ppm breakpoint range, assume µg/m³ and convert.
    if c.quantile(0.95) > 50.4:
        c = (c * 24.45) / (28.01 * 1000.0)  # µg/m³ -> ppm at 25C, 1 atm
    return c


def add_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add AQI overall using EPA breakpoints (max of pollutant sub-index)."""
    out = df.copy()

    pm25_bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    pm10_bp = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ]
    no2_bp = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ]
    so2_bp = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ]
    co_bp = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ]
    o3_bp = [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ]

    out["aqi_pm25"] = _aqi_subindex_series(out["PM2.5"], pm25_bp)
    out["aqi_pm10"] = _aqi_subindex_series(out["PM10"], pm10_bp)
    out["aqi_no2"] = _aqi_subindex_series(out["NO2"], no2_bp)
    out["aqi_so2"] = _aqi_subindex_series(out["SO2"], so2_bp)
    out["aqi_co"] = _aqi_subindex_series(_co_to_ppm_if_needed(out["CO"]), co_bp)
    out["aqi_o3"] = _aqi_subindex_series(out["O3"], o3_bp)

    out["AQI"] = out[["aqi_pm25", "aqi_pm10", "aqi_no2", "aqi_so2", "aqi_co", "aqi_o3"]].max(axis=1, skipna=True)
    return out


def aqi_category_from_value(aqi_value: float) -> str:
    if pd.isna(aqi_value):
        return "No Data"
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Moderate"
    if aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200:
        return "Unhealthy"
    if aqi_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def build_aqi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build AQI feature columns aligned with notebook logic."""
    out = add_aqi_column(df)
    out["aqi_category"] = out["AQI"].apply(aqi_category_from_value)
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["hour"] = out["datetime"].dt.hour
    out["season_label"] = out["month"].map(SEASON_MAP)

    # AQI category per pollutant sub-index for map and decomposition views.
    for pol, sub_col in SUBINDEX_MAP.items():
        out[f"{sub_col}_category"] = out[sub_col].apply(aqi_category_from_value)
    return out

# 3) DATA LOADING
@st.cache_data
def load_data(cache_version: str = CACHE_VERSION) -> pd.DataFrame:
    _ = cache_version  # cache bust key for data logic updates
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = ["datetime", "station"] + POLLUTANT_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak lengkap: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "station"]).copy()

    for col in POLLUTANT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = build_aqi_features(df)
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year

    return df

# FILTER COMPONENTS
def add_station_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "zone" in out.columns:
        zone_alias = {
            "Urban": "Urban",
            "Suburban": "Suburban",
            "Rural": "Rural",
            "Industrial": "Urban",
            "Perkotaan": "Urban",
            "Pinggiran": "Suburban",
            "Perdesaan": "Rural",
        }
        out["zone"] = out["zone"].astype("string").str.strip().replace(zone_alias)
        out["zone"] = out["zone"].fillna(out["station"].map(STATION_ZONE_MAP))
    else:
        out["zone"] = out["station"].map(STATION_ZONE_MAP)
    out["zone"] = out["zone"].fillna("Suburban")

    if "lat" not in out.columns:
        out["lat"] = out["station"].map(lambda s: STATION_COORDS.get(s, (np.nan, np.nan))[0])
    else:
        out["lat"] = out["lat"].fillna(out["station"].map(lambda s: STATION_COORDS.get(s, (np.nan, np.nan))[0]))

    if "lon" not in out.columns:
        out["lon"] = out["station"].map(lambda s: STATION_COORDS.get(s, (np.nan, np.nan))[1])
    else:
        out["lon"] = out["lon"].fillna(out["station"].map(lambda s: STATION_COORDS.get(s, (np.nan, np.nan))[1]))

    return out


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_station_priority(filtered: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "zone": "first",
        "lat": "median",
        "lon": "median",
        "PM2.5": "mean",
        "AQI": "mean",
    }
    for col in ["PM10", "NO2", "SO2", "CO", "O3", "TEMP", "WSPM"]:
        if col in filtered.columns:
            agg_dict[col] = "mean"

    station_priority = (
        filtered.groupby("station", as_index=False)
        .agg(agg_dict)
        .dropna(subset=["lat", "lon"], how="any")
    )

    gas_cols = [c for c in ["NO2", "SO2", "CO", "O3"] if c in station_priority.columns]
    if gas_cols:
        station_priority["gas_index"] = station_priority[gas_cols].mean(axis=1)
    else:
        station_priority["gas_index"] = 0.0

    station_priority["priority_score"] = (
        0.5 * _zscore(station_priority["PM2.5"])
        + 0.35 * _zscore(station_priority["AQI"])
        + 0.15 * _zscore(station_priority["gas_index"])
    )

    station_priority["aqi_category"] = station_priority["AQI"].apply(aqi_category_from_value)

    # Cluster disamakan dengan skala PM2.5 kuantil (Low/Medium/High).
    if len(station_priority) >= 3 and station_priority["PM2.5"].notna().sum() >= 3:
        q_low, q_high = station_priority["PM2.5"].quantile([1 / 3, 2 / 3])

        def pm25_cluster(v: float) -> str:
            if pd.isna(v):
                return "Medium"
            if v <= q_low:
                return "Low"
            if v <= q_high:
                return "Medium"
            return "High"

        station_priority["priority_cluster"] = station_priority["PM2.5"].apply(pm25_cluster)
    else:
        station_priority["priority_cluster"] = "Medium"

    return station_priority.sort_values("priority_score", ascending=False).reset_index(drop=True)


def render_q4_priority_map(station_priority: pd.DataFrame):
    st.markdown("### Peta Geospasial Area Prioritas")

    if not FOLIUM_READY:
        st.info("Instal `folium` dan `streamlit-folium` untuk menampilkan peta.")
        return

    if station_priority.empty:
        st.warning("Data stasiun tidak cukup untuk menampilkan peta prioritas.")
        return

    zone_colors = SOFT_ZONE_COLORS

    q_low, q_high = station_priority["PM2.5"].quantile([1 / 3, 2 / 3])

    def pm_radius_cluster(val: float) -> int:
        if pd.isna(val):
            return 6
        if val <= q_low:
            return 7
        if val <= q_high:
            return 10
        return 13

    center_lat = station_priority["lat"].mean()
    center_lon = station_priority["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")
    folium.TileLayer("CartoDB positron", name="CartoDB Positron").add_to(m)

    label_stations = set(
        station_priority.sort_values(["zone", "priority_score"], ascending=[True, False]).groupby("zone").head(3)["station"].tolist()
    )
    label_layer = folium.FeatureGroup(name="Labels", show=False)

    for _, r in station_priority.iterrows():
        aqi_val = r.get("AQI", np.nan)
        aqi_cat = r.get("aqi_category", "Unknown")
        temp_val = r.get("TEMP", np.nan)
        wind_val = r.get("WSPM", np.nan)

        temp_text = f"{temp_val:.1f}" if pd.notna(temp_val) else "NA"
        wind_text = f"{wind_val:.1f}" if pd.notna(wind_val) else "NA"
        tooltip_html = (
            f"{r['station']}<br>"
            f"Zone: {r['zone']}<br>"
            f"AQI: {aqi_val:.1f} ({aqi_cat})<br>"
            f"PM2.5: {r['PM2.5']:.1f} µg/m³<br>"
            f"Priority score: {r['priority_score']:.2f}<br>"
            f"Cluster: {r['priority_cluster']}<br>"
            f"Temp: {temp_text} C | Wind: {wind_text} m/s"
        )

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=pm_radius_cluster(r["PM2.5"]),
            color=zone_colors.get(r["zone"], SOFT_NEUTRAL_DARK),
            fill=True,
            fill_opacity=0.8,
            tooltip=tooltip_html,
        ).add_to(m)

        if r["station"] in label_stations:
            label_html = f'<div style="font-size:10px; color:#222; font-weight:bold;">{r["station"]}</div>'
            folium.Marker(location=[r["lat"], r["lon"]], icon=folium.DivIcon(html=label_html)).add_to(label_layer)

    label_layer.add_to(m)

    legend_html = (
        '<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
        'background: rgba(255,255,255,0.96); padding: 10px 12px; border: 1px solid #999; '
        'box-shadow: 0 2px 8px rgba(0,0,0,0.18); '
        'border-radius: 6px; font-size: 12px; line-height:1.35; color:#1f2d3d;">'
        "<strong>Zone</strong><br>"
        f'<span style="display:inline-block; width:10px; height:10px; background:{zone_colors["Urban"]}; margin-right:6px;"></span><span style="color:{zone_colors["Urban"]}; font-weight:600;">Urban</span><br>'
        f'<span style="display:inline-block; width:10px; height:10px; background:{zone_colors["Suburban"]}; margin-right:6px;"></span><span style="color:{zone_colors["Suburban"]}; font-weight:600;">Suburban</span><br>'
        f'<span style="display:inline-block; width:10px; height:10px; background:{zone_colors["Rural"]}; margin-right:6px;"></span><span style="color:{zone_colors["Rural"]}; font-weight:600;">Rural</span>'
        "</div>"
    )

    size_legend = (
        f'<div style="position: fixed; bottom: 24px; right: 24px; z-index: 9999; '
        f'background: rgba(255,255,255,0.96); padding: 10px 12px; border: 1px solid #999; '
        f'box-shadow: 0 2px 8px rgba(0,0,0,0.18); max-width: 210px; '
        f'border-radius: 6px; font-size: 12px; line-height:1.35; color:#1f2d3d;">'
        f"<strong>Skala PM2.5 (sesuai Cluster)</strong><br>"
        f'<span style="display:inline-block; width:12px; height:12px; border: 2px solid #333; border-radius: 50%;"></span>'
        f'<span style="margin-left:6px;">Low: <= {q_low:.1f}</span><br>'
        f'<span style="display:inline-block; width:16px; height:16px; border: 2px solid #333; border-radius: 50%;"></span>'
        f'<span style="margin-left:6px;">Medium: {q_low:.1f} - {q_high:.1f}</span><br>'
        f'<span style="display:inline-block; width:20px; height:20px; border: 2px solid #333; border-radius: 50%;"></span>'
        f'<span style="margin-left:6px;">High: > {q_high:.1f}</span>'
        f"</div>"
    )

    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(size_legend))
    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, use_container_width=True, height=540)


def render_q3_aqi_overall(filtered: pd.DataFrame):
    st.markdown("### Distribusi Kategori AQI Overall")
    if "aqi_category" not in filtered.columns or filtered["aqi_category"].dropna().empty:
        st.info("Data kategori AQI tidak cukup untuk visualisasi.")
        return

    cat_order = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
        "No Data",
    ]
    cat_series = pd.Categorical(filtered["aqi_category"], categories=cat_order, ordered=True)
    cat_counts = pd.Series(cat_series).value_counts(sort=False).reset_index()
    cat_counts.columns = ["aqi_category", "count"]
    cat_counts = cat_counts[cat_counts["count"] > 0]

    fig_dist = px.pie(
        cat_counts,
        names="aqi_category",
        values="count",
        color="aqi_category",
        color_discrete_map=AQI_COLOR_MAP_PASTEL_TEGAS,
        hole=0.35,
        template="plotly_white",
    )
    fig_dist.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="Kategori: %{label}<br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>",
    )
    fig_dist.update_layout(height=420)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### Persentase Kategori AQI Overall per Stasiun")
    station_cat = (
        filtered.groupby(["station", "aqi_category"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    if station_cat.empty:
        st.info("Data stasiun tidak cukup untuk visualisasi persentase kategori AQI.")
        return

    station_tot = station_cat.groupby("station")["count"].sum().rename("total").reset_index()
    station_cat = station_cat.merge(station_tot, on="station", how="left")
    station_cat["pct"] = np.where(station_cat["total"] > 0, station_cat["count"] / station_cat["total"] * 100.0, 0.0)
    station_cat["aqi_category"] = pd.Categorical(station_cat["aqi_category"], categories=cat_order, ordered=True)
    station_cat = station_cat.sort_values(["station", "aqi_category"])

    station_order = (
        filtered.groupby("station")["AQI"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig_station = px.bar(
        station_cat,
        x="station",
        y="pct",
        color="aqi_category",
        category_orders={"station": station_order, "aqi_category": cat_order},
        color_discrete_map=AQI_COLOR_MAP_PASTEL_TEGAS,
        template="plotly_white",
        barmode="stack",
    )
    fig_station.update_layout(
        yaxis_title="Persentase (%)",
        xaxis_title="Stasiun",
        height=460,
        legend_title_text="AQI Category",
    )
    fig_station.update_traces(hovertemplate="Stasiun: %{x}<br>Kategori: %{fullData.name}<br>Persentase: %{y:.1f}%<extra></extra>")
    st.plotly_chart(fig_station, use_container_width=True)


def render_best_worst_aqi_stations(filtered: pd.DataFrame):
    if "AQI" not in filtered.columns:
        return

    station_avg = (
        filtered.groupby("station", as_index=False)["AQI"]
        .mean()
        .dropna(subset=["AQI"])
        .sort_values("AQI", ascending=True)
    )
    if station_avg.empty:
        st.info("Data AQI stasiun tidak cukup untuk menampilkan 3 terbaik/terburuk.")
        return

    best3 = station_avg.head(3).copy()
    worst3 = station_avg.sort_values("AQI", ascending=False).head(3).copy()

    st.markdown("**3 Stasiun Terbaik (AQI terendah):**")
    st.dataframe(best3.round(2), use_container_width=True, hide_index=True)
    st.markdown("**3 Stasiun Terburuk (AQI tertinggi):**")
    st.dataframe(worst3.round(2), use_container_width=True, hide_index=True)


def render_zone_aqi_countplot(filtered: pd.DataFrame):
    st.markdown("### Countplot AQI category per zona")
    if "zone" not in filtered.columns or "aqi_category" not in filtered.columns:
        st.info("Data zona/kategori AQI belum tersedia untuk countplot.")
        return

    zone_order = ["Urban", "Suburban", "Rural"]
    cat_order = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
        "No Data",
    ]

    plot_df = (
        filtered.dropna(subset=["zone", "aqi_category"])
        .groupby(["zone", "aqi_category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if plot_df.empty:
        st.info("Data tidak cukup untuk menampilkan countplot AQI per zona.")
        return

    fig = px.bar(
        plot_df,
        x="zone",
        y="count",
        color="aqi_category",
        category_orders={"zone": zone_order, "aqi_category": cat_order},
        color_discrete_map=AQI_COLOR_MAP_PASTEL_TEGAS,
        barmode="group",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="Zona",
        yaxis_title="Jumlah Observasi",
        legend_title_text="AQI Category",
        height=420,
    )
    fig.update_traces(
        hovertemplate="Zona: %{x}<br>Kategori: %{fullData.name}<br>Jumlah: %{y}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_categorical_test_explanation():
    st.markdown("### Penjelasan Sederhana Uji Kategorikal (Chi-Square dan Countplot)")

    zone_aqi_tbl = pd.DataFrame(
        [
            {"zone": "Urban", "Good": 13935, "Hazardous": 9654, "Moderate": 50770, "Unhealthy": 105803, "Unhealthy for Sensitive Groups": 37818, "Very Unhealthy": 27468},
            {"zone": "Suburban", "Good": 6586, "Hazardous": 3165, "Moderate": 24711, "Unhealthy": 43128, "Unhealthy for Sensitive Groups": 16275, "Very Unhealthy": 11327},
            {"zone": "Rural", "Good": 7556, "Hazardous": 1419, "Moderate": 18403, "Unhealthy": 24483, "Unhealthy for Sensitive Groups": 11134, "Very Unhealthy": 7133},
        ]
    )
    season_pm_tbl = pd.DataFrame(
        [
            {"season": "AUTUMN", "Low": 35741, "Medium": 33072, "High": 36019},
            {"season": "SPRING", "Low": 30558, "Medium": 40914, "High": 34512},
            {"season": "SUMMER", "Low": 34133, "Medium": 43625, "High": 28226},
            {"season": "WINTER", "Low": 41119, "Medium": 22215, "High": 40634},
        ]
    )
    chi_summary = pd.DataFrame(
        [
            {"uji": "zone vs aqi_category", "chi2": 4600.685, "p_value": 0.0, "signif_5pct": "Ya"},
            {"uji": "season vs pm25_level_q", "chi2": 11819.360, "p_value": 0.0, "signif_5pct": "Ya"},
        ]
    )

    st.markdown("**Contingency table - zone vs aqi_category:**")
    st.dataframe(zone_aqi_tbl, use_container_width=True, hide_index=True)
    st.markdown("**Contingency table - season vs PM2.5 level (quantile):**")
    st.dataframe(season_pm_tbl, use_container_width=True, hide_index=True)
    st.markdown("**Ringkasan Chi-Square:**")
    st.dataframe(chi_summary, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Insight Uji Kategorikal (Chi-Square dan Countplot):**
- Hasil chi-square menunjukkan hubungan zona dengan kategori AQI itu nyata (signifikan), jadi kualitas udara memang berbeda antar urban, suburban, dan rural.
- Zona urban paling sering masuk kategori tidak sehat, jadi prioritas pengendalian emisi sebaiknya difokuskan ke area urban.
- Zona rural lebih banyak berada di kategori baik dan sedang, sehingga risikonya relatif lebih rendah.
- Hubungan musim dengan level PM2.5 juga signifikan, artinya risiko polusi berubah kuat antar musim.
- Musim dingin cenderung paling berisiko (PM2.5 tinggi), sedangkan musim panas relatif lebih rendah.
        """
    )


def render_filters(df: pd.DataFrame):
    min_date = pd.to_datetime(df["date"]).min().date()
    max_date = pd.to_datetime(df["date"]).max().date()

    # Date range filter
    date_range = st.sidebar.date_input(
        "Pilih Rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Station filter (multiselect)
    station_options = sorted(df["station"].dropna().unique().tolist())
    selected_stations = st.sidebar.multiselect(
        "Pilih Stasiun",
        options=station_options,
        default=station_options,
    )

    return start_date, end_date, selected_stations


def dominant_season_label(filtered: pd.DataFrame) -> str:
    season_id_map = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }

    if "season" in filtered.columns and filtered["season"].notna().any():
        season_val = filtered["season"].mode().iloc[0]
        if isinstance(season_val, (int, float, np.integer, np.floating)):
            return season_id_map.get(int(season_val), str(int(season_val)))
        return str(season_val)

    if filtered["datetime"].notna().any():
        month_val = int(filtered["datetime"].dt.month.mode().iloc[0])
        return month_to_season.get(month_val, "-")
    return "-"


def render_monthly_critical_period(filtered: pd.DataFrame, selected_pollutant: str):
    st.markdown(f"### 2. Periode Kritis Bulanan - {selected_pollutant}")

    f = filtered.copy()
    f["year_num"] = f["datetime"].dt.year
    f["month_num"] = f["datetime"].dt.month

    year_options = sorted(f["year_num"].dropna().unique().tolist())
    if not year_options:
        st.info("Data tahun tidak tersedia untuk heatmap bulanan.")
        return

    selected_year = st.selectbox("Pilih Tahun", options=year_options, index=len(year_options) - 1)

    yearly = f[f["year_num"] == selected_year].copy()
    if yearly.empty:
        st.warning("Tidak ada data untuk tahun terpilih.")
        return

    if selected_pollutant not in yearly.columns:
        st.warning(f"Kolom {selected_pollutant} tidak tersedia pada data.")
        return

    monthly_series = yearly.groupby("month_num")[selected_pollutant].mean()

    month_order = list(range(1, 13))
    monthly_series = monthly_series.reindex(month_order)
    month_labels = [f"{m:02d}" for m in month_order]
    heat_values = monthly_series.values.reshape(1, -1)

    zmax = np.nanpercentile(heat_values, 95) if np.isfinite(heat_values).any() else 1.0
    if not np.isfinite(zmax) or zmax <= 0:
        zmax = 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=heat_values,
            x=month_labels,
            y=[selected_pollutant],
            colorscale=SOFT_HEAT_COLORSCALE,
            zmin=0,
            zmax=float(zmax),
            colorbar=dict(title="µg/m³"),
            hovertemplate="Bulan %{x}<br>Polutan: %{y}<br>Rata-rata: %{z:.2f} µg/m³<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=290,
        xaxis_title="Bulan",
        yaxis_title="Polutan",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    if monthly_series.notna().any():
        peak_month = int(monthly_series.idxmax())
        peak_value = float(monthly_series.max())
        st.caption(f"Bulan puncak {selected_pollutant}: {peak_month:02d} ({peak_value:.2f} µg/m³).")


def render_daily_critical_pattern(filtered: pd.DataFrame, selected_pollutant: str):
    st.markdown(f"### 3. Pola Harian (Jam Kritis) - {selected_pollutant}")

    f = filtered.copy()
    f["hour_num"] = f["datetime"].dt.hour
    if selected_pollutant not in f.columns:
        st.info(f"Kolom {selected_pollutant} tidak tersedia untuk pola harian.")
        return
    hour_avg = f.groupby("hour_num")[selected_pollutant].mean().reindex(range(24))

    if hour_avg.dropna().empty:
        st.info("Data jam-an tidak cukup untuk menampilkan pola harian.")
        return

    peak_hour = int(hour_avg.idxmax())
    peak_value = float(hour_avg.max())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hour_avg.index,
            y=hour_avg.values,
            name=selected_pollutant,
            marker_color=SOFT_PRIMARY,
            opacity=0.7,
            hovertemplate="Jam %{x}:00<br>" + selected_pollutant + ": %{y:.2f} µg/m³<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak_hour],
            y=[peak_value],
            mode="markers+text",
            text=[f"Peak {selected_pollutant}"],
            textposition="top center",
            marker=dict(size=11, color=SOFT_NEUTRAL_DARK, symbol="diamond"),
            name=f"Jam kritis {selected_pollutant}",
            showlegend=False,
            hovertemplate=(
                f"{selected_pollutant} kritis jam {peak_hour:02d}:00"
                "<br>Rata-rata: %{y:.2f} µg/m³<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=410,
        xaxis=dict(title="Jam (0-23)", dtick=1),
        yaxis=dict(title="Rata-rata Konsentrasi (µg/m³)"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Jam puncak {selected_pollutant}: {peak_hour:02d}:00 ({peak_value:.2f} µg/m³).")


def render_pm_season_pattern(filtered: pd.DataFrame, selected_pollutant: str):
    st.markdown(f"### 4. Pola Musiman - {selected_pollutant}")

    f = filtered.copy()
    f["month_num"] = f["datetime"].dt.month
    # Hindari campuran dtype string vs float (np.nan) pada NumPy terbaru.
    f["season_pm"] = pd.Series(pd.NA, index=f.index, dtype="object")
    f.loc[f["month_num"].isin([6, 7, 8]), "season_pm"] = "Kemarau (Jun-Aug)"
    f.loc[f["month_num"].isin([12, 1, 2]), "season_pm"] = "Hujan (Des-Feb)"
    f = f[f["season_pm"].notna()].copy()

    if f.empty:
        st.info("Data musim kemarau/hujan tidak cukup untuk divisualkan.")
        return

    if selected_pollutant not in f.columns:
        st.info(f"Kolom {selected_pollutant} tidak tersedia untuk pola musiman.")
        return

    order = ["Kemarau (Jun-Aug)", "Hujan (Des-Feb)"]

    fig = go.Figure()
    for s, color in zip(order, [SOFT_SEASON_COLORS["Winter"], SOFT_SEASON_COLORS["Spring"]]):
        slice_df = f[f["season_pm"] == s]
        fig.add_trace(
            go.Box(
                x=[s] * len(slice_df),
                y=slice_df[selected_pollutant],
                name=s,
                marker_color=color,
                boxmean=True,
                hovertemplate=(
                    f"{selected_pollutant}<br>Musim: %{{x}}<br>Nilai: %{{y:.2f}} µg/m³<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=420,
        yaxis_title=f"{selected_pollutant} (µg/m³)",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    dry_mean = f.loc[f["season_pm"] == order[0], selected_pollutant].mean()
    wet_mean = f.loc[f["season_pm"] == order[1], selected_pollutant].mean()
    if pd.notna(dry_mean) and pd.notna(wet_mean):
        delta = dry_mean - wet_mean
        st.caption(
            f"Rata-rata {selected_pollutant} kemarau vs hujan: {dry_mean:.2f} vs {wet_mean:.2f} µg/m³ "
            f"(selisih {delta:+.2f})."
        )


def render_extreme_outlier_block(filtered: pd.DataFrame, selected_pollutant: str):
    st.markdown(f"### 5. Kejadian Ekstrem (Outlier) - {selected_pollutant}")

    pol = selected_pollutant
    out_df = filtered[["datetime", "year", "station", pol]].dropna().copy()
    if out_df.empty:
        st.info("Data tidak cukup untuk analisis outlier.")
        return

    bounds = (
        out_df.groupby("year")[pol]
        .agg(q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), mean_normal="mean")
        .reset_index()
    )
    bounds["iqr"] = bounds["q3"] - bounds["q1"]
    bounds["lb"] = bounds["q1"] - 1.5 * bounds["iqr"]
    bounds["ub"] = bounds["q3"] + 1.5 * bounds["iqr"]
    out_df = out_df.merge(bounds[["year", "lb", "ub", "mean_normal"]], on="year", how="left")
    out_df["is_outlier"] = (out_df[pol] < out_df["lb"]) | (out_df[pol] > out_df["ub"])

    fig = go.Figure()
    for y in sorted(out_df["year"].unique()):
        ydf = out_df[out_df["year"] == y]
        fig.add_trace(
            go.Box(
                x=[str(y)] * len(ydf),
                y=ydf[pol],
                name=str(y),
                boxpoints="outliers",
                marker=dict(color=SOFT_PRIMARY, outliercolor=SOFT_NEUTRAL_DARK, line=dict(outliercolor=SOFT_NEUTRAL_DARK, outlierwidth=1)),
                line=dict(color=SOFT_PRIMARY),
                fillcolor="rgba(126,168,199,0.18)",
                hovertemplate=f"Tahun {y}<br>{pol}: %{{y:.2f}} µg/m³<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=430,
        xaxis_title="Tahun",
        yaxis_title=f"{pol} (µg/m³)",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    selection = None
    try:
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            key="outlier_box_chart",
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

    outlier_only = out_df[out_df["is_outlier"]].copy().sort_values(pol, ascending=False)
    if outlier_only.empty:
        st.success("Tidak ada outlier ekstrem berdasarkan aturan IQR pada filter aktif.")
        return

    selected_row = None
    if selection is not None and hasattr(selection, "selection"):
        sel_payload = selection.selection if isinstance(selection.selection, dict) else {}
        points = sel_payload.get("points", [])
        if points:
            p = points[0]
            year_clicked = int(float(p.get("x")))
            value_clicked = float(p.get("y"))
            cand = outlier_only[outlier_only["year"] == year_clicked].copy()
            if not cand.empty:
                cand["dist"] = (cand[pol] - value_clicked).abs()
                selected_row = cand.sort_values("dist").iloc[0]

    if selected_row is None:
        st.caption("Klik/seleksi titik outlier pada boxplot untuk melihat detail. Fallback detail otomatis ditampilkan di bawah.")
        selected_row = outlier_only.iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Tanggal kejadian", selected_row["datetime"].strftime("%Y-%m-%d %H:%M"))
    with c2:
        st.metric(f"Nilai {pol}", f"{selected_row[pol]:.2f} µg/m³")


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return np.nan
    if pair["x"].nunique() <= 1 or pair["y"].nunique() <= 1:
        return np.nan
    return pair["x"].corr(pair["y"], method="spearman")


def _build_season_corr(df: pd.DataFrame, target_col: str, met_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=met_cols, columns=SEASON_ORDER, dtype="float64")
    for season in SEASON_ORDER:
        s_df = df[df["season_label"] == season]
        for met in met_cols:
            out.loc[met, season] = _safe_spearman(s_df[target_col], s_df[met])
    return out


def _build_peak_month_hour_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pol in POLLUTANT_COLS:
        by_month = df.groupby(df["datetime"].dt.month)[pol].mean()
        by_hour = df.groupby(df["datetime"].dt.hour)[pol].mean()
        if by_month.dropna().empty or by_hour.dropna().empty:
            rows.append({"Polutan": pol, "Bulan Puncak": "-", "Jam Puncak": "-"})
            continue
        rows.append(
            {
                "Polutan": pol,
                "Bulan Puncak": int(by_month.idxmax()),
                "Jam Puncak": int(by_hour.idxmax()),
            }
        )
    return pd.DataFrame(rows)


def render_q1_dynamic_insight(filtered: pd.DataFrame, metric_x: str, metric_unit: str):
    if metric_x not in filtered.columns:
        st.info(f"Metric `{metric_x}` tidak tersedia untuk insight.")
        return

    f = filtered.copy()
    f["year_num"] = f["datetime"].dt.year
    f["month_num"] = f["datetime"].dt.month
    f["hour_num"] = f["datetime"].dt.hour
    f[metric_x] = pd.to_numeric(f[metric_x], errors="coerce")

    def month_label(m: int) -> str:
        if pd.isna(m):
            return "-"
        return ID_MONTH.get(int(m), str(int(m)))

    # 1) Trend akhir periode (sesuai metric aktif)
    trend_text = f"Data tahunan `{metric_x}` belum cukup untuk menilai tren akhir periode."
    yearly = f.groupby("year_num")[metric_x].mean().dropna()
    if len(yearly) >= 2:
        y0, y1 = int(yearly.index.min()), int(yearly.index.max())
        v0, v1 = float(yearly.loc[y0]), float(yearly.loc[y1])
        if v1 > v0:
            condition = "memburuk (naik)"
        elif v1 < v0:
            condition = "membaik (turun)"
        else:
            condition = "stabil"
        trend_text = (
            f"Tren {y0}-{y1} untuk `{metric_x}` menunjukkan kondisi **{condition}**, "
            f"dari {v0:.1f} ke {v1:.1f} {metric_unit}."
        )

    # 2) Periode kritis bulanan (sesuai metric aktif)
    monthly_text = f"Data bulanan `{metric_x}` belum cukup."
    by_month = f.groupby("month_num")[metric_x].mean().dropna()
    if not by_month.empty:
        peak_month = int(by_month.idxmax())
        peak_month_val = float(by_month.max())
        monthly_text = (
            f"Puncak bulanan `{metric_x}` terjadi pada **{month_label(peak_month)}** "
            f"dengan rerata {peak_month_val:.2f} {metric_unit}."
        )

    # 3) Pola musiman 4 musim untuk metric aktif
    seasonal_text = f"Data 4 musim `{metric_x}` belum cukup."
    season_col = f["season_label"] if "season_label" in f.columns else f["month_num"].map(SEASON_MAP)
    seasonal_avg = (
        f.assign(_season=season_col)
        .groupby("_season")[metric_x]
        .mean()
        .reindex(["Winter", "Spring", "Summer", "Autumn"])
        .dropna()
    )
    if len(seasonal_avg) >= 2:
        season_id = {
            "Winter": "Winter",
            "Spring": "Spring",
            "Summer": "Summer",
            "Autumn": "Autumn",
        }
        detail = ", ".join(f"{season_id.get(k, k)} {v:.2f}" for k, v in seasonal_avg.items())
        peak_season = seasonal_avg.idxmax()
        low_season = seasonal_avg.idxmin()
        seasonal_text = (
            f"Rerata `{metric_x}` per musim: {detail} {metric_unit}. "
            f"Tertinggi di **{season_id.get(peak_season, peak_season)}** "
            f"({seasonal_avg.max():.2f}) dan terendah di "
            f"**{season_id.get(low_season, low_season)}** ({seasonal_avg.min():.2f})."
        )

    # 4) Pola harian (jam kritis) untuk metric aktif
    hourly_text = f"Data jam-an `{metric_x}` belum cukup."
    by_hour = f.groupby("hour_num")[metric_x].mean().dropna()
    if not by_hour.empty:
        peak_hour = int(by_hour.idxmax())
        peak_hour_val = float(by_hour.max())
        hourly_text = (
            f"Jam puncak `{metric_x}` terjadi sekitar **{peak_hour:02d}.00** "
            f"dengan rerata {peak_hour_val:.2f} {metric_unit}."
        )

    # 5) Outlier untuk metric aktif
    outlier_text = f"Belum ada outlier ekstrem `{metric_x}` yang menonjol pada filter saat ini."
    s = f[metric_x].dropna()
    if len(s) >= 4:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        ub = q3 + 1.5 * iqr
        out = f[f[metric_x] > ub].sort_values(metric_x, ascending=False)
        if not out.empty:
            top = out.iloc[0]
            dt_txt = pd.to_datetime(top["datetime"]).strftime("%Y-%m-%d %H:%M")
            outlier_text = (
                f"Terdeteksi {len(out):,} outlier `{metric_x}`; puncak ekstrem terjadi pada {dt_txt} "
                f"dengan nilai {float(top[metric_x]):.2f} {metric_unit}. "
                "Perlu ditelusuri faktor pemicu (cuaca, emisi, dan aktivitas lokal)."
            )

    st.markdown(
        f"""
**Insight Tren dan Waktu Kritis - Filter Aktif:**
- **Tren Akhir Periode:** {trend_text}
- **Periode Kritis Bulanan:** {monthly_text}
- **Pola Musiman:** {seasonal_text}
- **Pola Harian (Jam Kritis):** {hourly_text}
- **Kejadian Ekstrem (Outlier):** {outlier_text}
        """
    )


def render_q2_meteorology_section(filtered: pd.DataFrame, metric_x: str):
    if metric_x not in filtered.columns:
        st.info(f"Metric `{metric_x}` tidak tersedia untuk analisis.")
        return

    metric_unit = "µg/m³" if metric_x in POLLUTANT_COLS else "AQI index"
    st.markdown(f"### Meteorologi dan Risiko Polusi ({metric_x})")

    q2_cols = ["datetime", "season_label", "wd", "AQI", metric_x, "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    # Hindari duplikasi nama kolom (contoh: metric_x == "AQI").
    q2_cols_unique = list(dict.fromkeys(c for c in q2_cols if c in filtered.columns))
    qdf = filtered[q2_cols_unique].dropna(subset=[metric_x]).copy()
    if qdf.empty:
        st.info("Data tidak cukup untuk analisis pada filter saat ini.")
        return

    qdf["month_num"] = qdf["datetime"].dt.month
    qdf["month_label"] = qdf["datetime"].dt.strftime("%b")
    month_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    month_label_order = [pd.Timestamp(2000, m, 1).strftime("%b") for m in month_order]

    st.subheader(f"Season Averages {metric_x}")
    seasonal_avg = qdf.groupby("season_label")[metric_x].mean().reindex(SEASON_ORDER)

    st.markdown(
        """
        <style>
          .q2-season-card {
            background: #f7f9fc;
            border: 1px solid #e3e9f3;
            border-radius: 12px;
            padding: 12px;
            min-height: 96px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
          }
          .q2-season-label {
            font-size: 12px;
            color: #5f6d7a;
            font-weight: 700;
            line-height: 1.2;
          }
          .q2-season-value {
            margin-top: 8px;
            font-size: 28px;
            color: #1f2d3d;
            font-weight: 800;
            line-height: 1.1;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    season_cols = [c1, c2, c3, c4]
    season_names = ["Winter", "Spring", "Summer", "Autumn"]
    for col, season in zip(season_cols, season_names):
        val = seasonal_avg.get(season, np.nan)
        val_display = f"{val:.2f}" if pd.notna(val) else "-"
        col.markdown(
            f"""
            <div class="q2-season-card">
              <div class="q2-season-label">{season} avg {metric_x} konsentrasi</div>
              <div class="q2-season-value">{val_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f"#### Seasonal {metric_x} Distribution (Subplots)")
    fig_season = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Winter", "Spring", "Summer", "Autumn"],
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )
    season_pos = {"Winter": (1, 1), "Spring": (1, 2), "Summer": (2, 1), "Autumn": (2, 2)}
    season_colors = SOFT_SEASON_COLORS
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        rdf = qdf[qdf["season_label"] == season]
        r, c = season_pos[season]
        fig_season.add_trace(
            go.Box(
                y=rdf[metric_x],
                name=season,
                marker_color=season_colors[season],
                boxmean=True,
                showlegend=False,
                hovertemplate=f"Musim: {season}<br>{metric_x}: %{{y:.2f}} {metric_unit}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    fig_season.update_layout(template="plotly_white", height=520)
    fig_season.update_yaxes(title_text=f"{metric_x} ({metric_unit})")
    st.plotly_chart(fig_season, use_container_width=True)

    if metric_x != "AQI":
        st.markdown(f"#### Monthly {metric_x} Distribution (Violinplot)")
        fig_violin = px.violin(
            qdf,
            x="month_label",
            y=metric_x,
            category_orders={"month_label": month_label_order},
            box=True,
            points=False,
            template="plotly_white",
            title=f"Distribusi Bulanan {metric_x}",
        )
        fig_violin.update_layout(xaxis_title="Bulan", yaxis_title=f"{metric_x} ({metric_unit})")
        st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown(f"#### Korelasi Musiman {metric_x} dan Meteorologi")
    met_cols = [c for c in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"] if c in qdf.columns]
    corr = pd.DataFrame(index=met_cols, columns=["Winter", "Spring", "Summer", "Autumn"], dtype="float64")
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        sdf = qdf[qdf["season_label"] == season]
        for met in met_cols:
            corr.loc[met, season] = _safe_spearman(sdf[metric_x], sdf[met])
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale=SOFT_DIVERGING_COLORSCALE,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Spearman"),
            hovertemplate="Musim %{x}<br>Variabel %{y}<br>Korelasi: %{z:.3f}<extra></extra>",
        )
    )
    fig_corr.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown(f"#### Arah Angin (wd) dan Rerata {metric_x} per Musim")
    wind_rows = []
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        sdf = qdf[qdf["season_label"] == season]
        top = sdf.groupby("wd")[metric_x].mean().sort_values(ascending=False).head(3)
        for wd_code, val in top.items():
            wind_rows.append({"season": season, "wd": wd_code, "val": val})
    wind_df = pd.DataFrame(wind_rows)
    if not wind_df.empty:
        fig_wind = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Winter", "Spring", "Summer", "Autumn"],
            horizontal_spacing=0.08,
            vertical_spacing=0.18,
        )
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            r, c = season_pos[season]
            sdf = wind_df[wind_df["season"] == season]
            fig_wind.add_trace(
                go.Bar(
                    x=sdf["wd"],
                    y=sdf["val"],
                    marker_color=season_colors[season],
                    showlegend=False,
                    hovertemplate=f"Musim: {season}<br>wd: %{{x}}<br>Rerata {metric_x}: %{{y:.2f}} {metric_unit}<extra></extra>",
                ),
                row=r,
                col=c,
            )
        fig_wind.update_layout(template="plotly_white", height=560)
        fig_wind.update_yaxes(title_text=f"{metric_x} ({metric_unit})")
        fig_wind.update_xaxes(title_text="Arah Angin (wd)")
        st.plotly_chart(fig_wind, use_container_width=True)
    else:
        st.info("Data arah angin tidak cukup untuk divisualisasikan.")

    def corr_label(r: float) -> str:
        if pd.isna(r):
            return "tidak cukup data"
        direction = "positif" if r > 0 else "negatif" if r < 0 else "netral"
        ar = abs(r)
        if ar >= 0.7:
            strength = "sangat kuat"
        elif ar >= 0.5:
            strength = "kuat"
        elif ar >= 0.3:
            strength = "sedang"
        elif ar >= 0.1:
            strength = "lemah"
        else:
            strength = "sangat lemah"
        return f"{direction} ({strength}, r={r:.2f})"

    def strongest_season(var_name: str):
        if var_name not in corr.index:
            return "-", np.nan
        row = corr.loc[var_name].dropna()
        if row.empty:
            return "-", np.nan
        season = row.abs().idxmax()
        return season, float(row.loc[season])

    dewp_r = _safe_spearman(qdf[metric_x], qdf["DEWP"]) if "DEWP" in qdf.columns else np.nan
    wspm_r = _safe_spearman(qdf[metric_x], qdf["WSPM"]) if "WSPM" in qdf.columns else np.nan
    temp_r = _safe_spearman(qdf[metric_x], qdf["TEMP"]) if "TEMP" in qdf.columns else np.nan
    pres_r = _safe_spearman(qdf[metric_x], qdf["PRES"]) if "PRES" in qdf.columns else np.nan
    rain_r = _safe_spearman(qdf[metric_x], qdf["RAIN"]) if "RAIN" in qdf.columns else np.nan

    dewp_peak_season, dewp_peak_val = strongest_season("DEWP")
    wspm_peak_season, wspm_peak_val = strongest_season("WSPM")
    temp_peak_season, temp_peak_val = strongest_season("TEMP")
    pres_peak_season, pres_peak_val = strongest_season("PRES")
    rain_peak_season, rain_peak_val = strongest_season("RAIN")

    winter_wd = "-"
    summer_wd = "-"
    if not wind_df.empty:
        winter_top = wind_df[wind_df["season"] == "Winter"].sort_values("val", ascending=False)
        summer_top = wind_df[wind_df["season"] == "Summer"].sort_values("val", ascending=False)
        if not winter_top.empty:
            winter_wd = str(winter_top.iloc[0]["wd"])
        if not summer_top.empty:
            summer_wd = str(summer_top.iloc[0]["wd"])

    st.markdown(
        f"""
**Insight Meteorologi dan Risiko Polusi - Filter Aktif:**
- **DEWP (Titik Embun) sebagai Faktor Penjerat Polusi:** Untuk **{metric_x}**, hubungan dengan DEWP bersifat **{corr_label(dewp_r)}**. Musim dengan pengaruh paling menonjol: **{dewp_peak_season}** (r={dewp_peak_val:.2f} jika tersedia).
- **WSPM (Kecepatan Angin) sebagai Faktor Dispersi:** Hubungan **{metric_x} vs WSPM**: **{corr_label(wspm_r)}**. Jika korelasi negatif, ini menandakan angin membantu menyebarkan polutan. Musim terkuat: **{wspm_peak_season}** (r={wspm_peak_val:.2f} jika tersedia).
- **Suhu (TEMP) Memiliki Efek Berlawanan:** Hubungan **{metric_x} vs TEMP** saat ini: **{corr_label(temp_r)}**. Musim dengan pola suhu paling dominan: **{temp_peak_season}** (r={temp_peak_val:.2f} jika tersedia). Ini membantu menentukan fokus pengawasan per musim.
- **Tekanan Tinggi (PRES) Memperparah, Hujan (RAIN) Membersihkan:** Korelasi **{metric_x} vs PRES**: **{corr_label(pres_r)}** (musim terkuat: **{pres_peak_season}**). Korelasi **{metric_x} vs RAIN**: **{corr_label(rain_r)}** (musim terkuat: **{rain_peak_season}**). Pola ini berguna untuk kewaspadaan saat tekanan tinggi atau saat hujan minim.
- **Arah Angin Menentukan Sumber Polusi:** Berdasarkan filter aktif, arah angin yang paling sering terkait nilai tinggi **{metric_x}**: **Winter = {winter_wd}**, **Summer = {summer_wd}**. Arah ini dapat dipakai sebagai indikator kemungkinan sumber emisi dominan.
        """
    )

def main():
    st.title("🌍 Beijing Air Quality Dashboard (2013-2017)")

    try:
        df = load_data()
        df = add_station_context(df)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

    start_date, end_date, selected_stations = render_filters(df)
    if start_date > end_date:
        st.error("Tanggal mulai tidak boleh lebih besar dari tanggal selesai.")
        st.stop()

    filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if selected_stations:
        filtered = filtered[filtered["station"].isin(selected_stations)]

    if filtered.empty:
        st.warning("Tidak ada data untuk kombinasi filter saat ini.")
        st.stop()

    # Header cards: AQI, dominant pollutant, and weather context.

    current_aqi = float(filtered["AQI"].mean()) if filtered["AQI"].notna().any() else np.nan
    aqi_category = aqi_category_from_value(current_aqi)
    aqi_color = AQI_COLOR_MAP.get(aqi_category, "#9E9E9E")
    health_impact = AQI_HEALTH_IMPACT.get(aqi_category, "-")

    dominant_pollutant = max(
        SUBINDEX_MAP.keys(),
        key=lambda p: filtered[SUBINDEX_MAP[p]].mean(skipna=True) if SUBINDEX_MAP[p] in filtered.columns else -np.inf,
    )
    dominant_conc = filtered[dominant_pollutant].mean(skipna=True) if dominant_pollutant in filtered.columns else np.nan
    dominant_aqi = (
        filtered[SUBINDEX_MAP[dominant_pollutant]].mean(skipna=True)
        if SUBINDEX_MAP[dominant_pollutant] in filtered.columns
        else np.nan
    )
    dominant_aqi_category = aqi_category_from_value(dominant_aqi)

    pollutant_aqi_stats = {}
    for pol, sub_col in SUBINDEX_MAP.items():
        sub_val = filtered[sub_col].mean(skipna=True) if sub_col in filtered.columns else np.nan
        pollutant_aqi_stats[pol] = {
            "aqi": sub_val,
            "category": aqi_category_from_value(sub_val),
        }

    # Kontribusi polutan berdasarkan porsi "dominant pollutant" pada periode terfilter.
    pol_order = ["PM2.5", "PM10", "O3", "CO", "NO2", "SO2"]
    pol_to_sub = {p: SUBINDEX_MAP[p] for p in pol_order if p in SUBINDEX_MAP}
    sub_cols = [c for c in pol_to_sub.values() if c in filtered.columns]
    contribution_pct = {p: 0.0 for p in pol_order}
    if sub_cols:
        risk_scope = filtered.dropna(subset=sub_cols, how="all").copy()
        if not risk_scope.empty:
            dominant_sub = risk_scope[sub_cols].idxmax(axis=1)
            counts = dominant_sub.value_counts()
            total = int(counts.sum())
            if total > 0:
                inv_map = {v: k for k, v in pol_to_sub.items()}
                for sub_col, cnt in counts.items():
                    pol = inv_map.get(sub_col)
                    if pol:
                        contribution_pct[pol] = float(cnt) / float(total) * 100.0

    # Dominant weather profile over selected date range.
    temp_dom = filtered["TEMP"].median(skipna=True) if "TEMP" in filtered.columns else np.nan
    pres_dom = filtered["PRES"].median(skipna=True) if "PRES" in filtered.columns else np.nan
    dewp_dom = filtered["DEWP"].median(skipna=True) if "DEWP" in filtered.columns else np.nan
    rain_dom = filtered["RAIN"].median(skipna=True) if "RAIN" in filtered.columns else np.nan
    wind_speed_dom = filtered["WSPM"].median(skipna=True) if "WSPM" in filtered.columns else np.nan
    wind_dir_dom = filtered["wd"].mode().iloc[0] if "wd" in filtered.columns and filtered["wd"].notna().any() else "-"

    aqi_display = f"{current_aqi:.0f}" if pd.notna(current_aqi) else "-"
    dom_conc_display = f"{dominant_conc:.2f} µg/m³" if pd.notna(dominant_conc) else "- µg/m³"
    dom_aqi_display = f"{dominant_aqi:.1f}" if pd.notna(dominant_aqi) else "-"
    temp_display = f"{temp_dom:.1f} °C" if pd.notna(temp_dom) else "- °C"
    pres_display = f"{pres_dom:.1f} hPa" if pd.notna(pres_dom) else "- hPa"
    dewp_display = f"{dewp_dom:.1f} °C" if pd.notna(dewp_dom) else "- °C"
    rain_display = f"{rain_dom:.2f} mm" if pd.notna(rain_dom) else "- mm"
    wind_speed_display = f"{wind_speed_dom:.2f} m/s" if pd.notna(wind_speed_dom) else "- m/s"
    start_date_text = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_text = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    dominant_range_text = start_date_text if start_date_text == end_date_text else f"{start_date_text} s.d. {end_date_text}"
    dominant_contrib_display = f"{contribution_pct.get(dominant_pollutant, 0.0):.1f}%"

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.0rem; }
          .header-card {
            background: #f7f9fc;
            border: 1px solid #e3e9f3;
            border-radius: 12px;
            padding: 12px;
            min-height: 184px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
          }
          .header-label {
            font-size: 11px;
            color: #5f6d7a;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
          }
          .header-main {
            color: #1f2d3d;
            font-weight: 800;
            line-height: 1.2;
          }
          .header-muted {
            font-size: 12px;
            color: #4f5b6a;
            line-height: 1.35;
          }
          .quick-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-top: 6px;
            margin-bottom: 6px;
          }
          .chip {
            background: #eef3fa;
            border: 1px solid #d8e2f0;
            border-radius: 8px;
            padding: 6px 8px;
            font-size: 11px;
            color: #2c3e50;
            font-weight: 600;
            line-height: 1.2;
          }
          .pollutant-card {
            background: #f7f9fc;
            border: 1px solid #e3e9f3;
            border-radius: 10px;
            padding: 10px;
            min-height: 92px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
          }
          .pollutant-name { font-size: 13px; font-weight: 800; color: #1f2d3d; }
          .pollutant-desc { font-size: 10px; color: #6b778c; line-height: 1.25; }
          .pollutant-val { font-size: 15px; font-weight: 800; color: #1f2d3d; }
          .pollutant-aqi { font-size: 11px; color: #4f5b6a; margin-top: 4px; }
          .pollutant-cat { font-size: 11px; color: #1f2d3d; font-weight: 700; }
          .pollutant-contrib { font-size: 11px; color: #2c3e50; margin-top: 4px; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns([1.05, 1.15, 1.35])
    with h1:
        st.markdown(
            f"""
            <div class="header-card">
              <div class="header-label">🫁 AQI Rentang Filter</div>
              <div style="font-size:42px;color:{aqi_color};font-weight:800;line-height:1.02;">{aqi_display}</div>
              <div style="font-size:13px;font-weight:800;color:#1f2d3d;">{aqi_category}</div>
              <div class="header-muted">{health_impact}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            f"""
            <div class="header-card">
              <div class="header-label">🏭 Polutan Dominan ({dominant_range_text})</div>
              <div style="font-size:24px;" class="header-main">{dominant_pollutant}</div>
              <div class="header-label">Konsentrasi</div>
              <div style="font-size:20px;" class="header-main">{dom_conc_display}</div>
              <div class="header-muted">AQI {dominant_pollutant}: <b>{dom_aqi_display}</b></div>
              <div class="header-muted">Kategori AQI {dominant_pollutant}: <b>{dominant_aqi_category}</b></div>
              <div class="header-muted">Kontribusi {dominant_pollutant}: <b>{dominant_contrib_display}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h3:
        st.markdown(
            f"""
            <div class="header-card">
              <div class="header-label">🌤️ Cuaca Dominan ({dominant_range_text})</div>
              <div class="quick-grid">
                <div class="chip">🌡️ TEMP: {temp_display}</div>
                <div class="chip">🧭 PRES: {pres_display}</div>
                <div class="chip">💧 DEWP: {dewp_display}</div>
                <div class="chip">🌧️ RAIN: {rain_display}</div>
                <div class="chip">🗺️ Arah Angin: {wind_dir_dom}</div>
                <div class="chip">💨 WSPM: {wind_speed_display}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(" ")
    pollutant_meta_all = [
        ("PM2.5", "Fine particle (=< 2.5 µg)"),
        ("PM10", "Coarse particles (=< 10 µg)"),
        ("O3", "Ozone"),
        ("NO2", "Nitrogen Dioxide"),
        ("SO2", "Sulphur Dioxide"),
        ("CO", "Carbon Monoxide"),
    ]
    pollutant_meta = [(pol, desc) for pol, desc in pollutant_meta_all if pol != dominant_pollutant]
    pcols = st.columns(len(pollutant_meta))
    for col_ui, (pol, desc) in zip(pcols, pollutant_meta):
        pol_val = filtered[pol].mean(skipna=True) if pol in filtered.columns else np.nan
        pol_display = f"{pol_val:.2f} µg/m³" if pd.notna(pol_val) else "- µg/m³"
        pol_aqi = pollutant_aqi_stats.get(pol, {}).get("aqi", np.nan)
        pol_aqi_display = f"{pol_aqi:.1f}" if pd.notna(pol_aqi) else "-"
        pol_aqi_cat = pollutant_aqi_stats.get(pol, {}).get("category", "No Data")
        pol_contrib_display = f"{contribution_pct.get(pol, 0.0):.1f}%"
        with col_ui:
            st.markdown(
                f"""
                <div class="pollutant-card">
                  <div class="pollutant-name">{pol}</div>
                  <div class="pollutant-desc">{desc}</div>
                  <div class="pollutant-val">{pol_display}</div>
                  <div class="pollutant-aqi">AQI {pol}: {pol_aqi_display}</div>
                  <div class="pollutant-cat">Kategori AQI: {pol_aqi_cat}</div>
                  <div class="pollutant-contrib">Kontribusi: {pol_contrib_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    current_history_metric = st.session_state.get("history_metric", "AQI")
    metric_idx = HISTORY_METRICS.index(current_history_metric) if current_history_metric in HISTORY_METRICS else 0
    selected_metric = st.sidebar.selectbox(
        "Metric Air Polutant",
        options=HISTORY_METRICS,
        index=metric_idx,
        key="history_metric",
    )

    current_view_mode = st.session_state.get("view_mode", "Harian")
    st.markdown(f"### History Trend ({current_view_mode}) - ({selected_metric})")
    view_mode = st.radio("Tampilan", options=["Harian", "Bulanan", "Tahunan"], horizontal=True, key="view_mode")

    if view_mode == "Harian":
        history = (
            filtered.assign(period=filtered["datetime"].dt.floor("D"))
            .groupby("period", as_index=False)
            .agg(value=(selected_metric, "mean"), aqi_mean=("AQI", "mean"))
            .sort_values("period")
        )
        x_col = "period"
        x_label = "Tanggal"
    elif view_mode == "Bulanan":
        history = (
            filtered.assign(period=filtered["datetime"].dt.to_period("M").dt.to_timestamp())
            .groupby("period", as_index=False)
            .agg(value=(selected_metric, "mean"), aqi_mean=("AQI", "mean"))
            .sort_values("period")
        )
        x_col = "period"
        x_label = "Bulan"
    else:
        history = (
            filtered.groupby("year", as_index=False)
            .agg(value=(selected_metric, "mean"), aqi_mean=("AQI", "mean"))
            .sort_values("year")
            .rename(columns={"year": "period"})
        )
        x_col = "period"
        x_label = "Tahun"

    history["aqi_category"] = history["aqi_mean"].apply(aqi_category_from_value)
    metric_unit_map = {
        "AQI": "AQI index",
        "PM2.5": "µg/m³",
        "PM10": "µg/m³",
        "O3": "µg/m³",
        "CO": "µg/m³",
        "NO2": "µg/m³",
        "SO2": "µg/m³",
    }
    selected_unit = metric_unit_map.get(selected_metric, "-")

    fig = px.line(
        history,
        x=x_col,
        y="value",
        labels={x_col: x_label, "value": selected_metric},
    )
    fig.update_traces(
        line=dict(width=2, color=SOFT_PRIMARY),
        customdata=np.stack([history["aqi_mean"].round(1), history["aqi_category"]], axis=-1),
        hovertemplate=(
            f"{x_label}: %{{x}}<br>"
            f"{selected_metric}: %{{y:.2f}}<br>"
            f"Satuan: {selected_unit}<br>"
            "AQI (avg): %{customdata[0]}<br>"
            "AQI Category: %{customdata[1]}<extra></extra>"
        ),
    )
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Visualisasi Heatmap Pola Jam-Bulan ({selected_metric})")
    c_heat, c_tbl = st.columns([2.4, 1.6])

    metric_heat = (
        filtered.groupby([filtered["datetime"].dt.hour, filtered["datetime"].dt.month])[selected_metric]
        .mean()
        .unstack()
        .reindex(index=range(24), columns=range(1, 13))
    )
    by_month_metric = filtered.groupby(filtered["datetime"].dt.month)[selected_metric].mean()
    by_hour_metric = filtered.groupby(filtered["datetime"].dt.hour)[selected_metric].mean()

    with c_heat:
        fig_metric_heat = go.Figure(
            data=go.Heatmap(
                z=metric_heat.values,
                x=[str(i) for i in range(1, 13)],
                y=[str(i) for i in range(0, 24)],
                colorscale=SOFT_HEAT_COLORSCALE,
                colorbar=dict(title=selected_unit),
                hovertemplate=(
                    "Bulan %{x}<br>Jam %{y}:00<br>"
                    + selected_metric
                    + ": %{z:.2f} "
                    + selected_unit
                    + "<extra></extra>"
                ),
            )
        )
        fig_metric_heat.update_layout(
            template="plotly_white",
            xaxis_title="Bulan",
            yaxis_title="Jam",
            height=430,
        )
        st.plotly_chart(fig_metric_heat, use_container_width=True)

    with c_tbl:
        st.markdown("**Tabel Ringkas (Summary Table)**")
        if by_month_metric.dropna().empty or by_hour_metric.dropna().empty:
            summary_df = pd.DataFrame(
                [{"Metric": selected_metric, "Bulan Puncak": "-", "Jam Puncak": "-", "Nilai Puncak": "-"}]
            )
        else:
            peak_month = int(by_month_metric.idxmax())
            peak_hour = int(by_hour_metric.idxmax())
            peak_value = float(by_hour_metric.max())
            summary_df = pd.DataFrame(
                [
                    {
                        "Metric": selected_metric,
                        "Bulan Puncak": peak_month,
                        "Jam Puncak": peak_hour,
                        "Nilai Puncak": f"{peak_value:.2f} {selected_unit}",
                    }
                ]
            )
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="small"),
                "Bulan Puncak": st.column_config.NumberColumn("Bulan Puncak", width="small"),
                "Jam Puncak": st.column_config.NumberColumn("Jam Puncak", width="small"),
                "Nilai Puncak": st.column_config.TextColumn("Nilai Puncak", width="medium"),
            },
        )
        
    render_q1_dynamic_insight(filtered, selected_metric, selected_unit)

    st.markdown("---")
    render_q2_meteorology_section(filtered, selected_metric)
    st.markdown("---")
    if selected_metric == "AQI":
        render_q3_aqi_overall(filtered)
        st.markdown("---")
        render_best_worst_aqi_stations(filtered)
        st.markdown("---")
    station_priority = build_station_priority(filtered)
    render_q4_priority_map(station_priority)
    if selected_metric == "AQI":
        st.markdown("---")
        render_zone_aqi_countplot(filtered)
        st.markdown("---")
        render_categorical_test_explanation()

    with st.expander("Lihat data hasil filter"):
        show_cols = ["datetime", "station", "AQI", "PM2.5", "PM10", "O3", "CO", "NO2", "SO2"]
        existing_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[existing_cols].sort_values("datetime", ascending=False), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

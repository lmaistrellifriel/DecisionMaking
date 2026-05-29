import math
import re
import time
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="WTG Main Component - Cost Minimizer (Advanced DSS)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# CONSTANTS / LABELS
# ==============================
STATE_NONE = 0
STATE_WORK = 1
STATE_STANDBY = 2
STATE_STOP_NIGHT = 3
STATE_WAIT_NO_CRANE = 4

STATE_LABELS = {
    STATE_NONE: "Fuori cantiere",
    STATE_WORK: "Lavoro",
    STATE_STANDBY: "Standby",
    STATE_STOP_NIGHT: "Stop Notte",
    STATE_WAIT_NO_CRANE: "Attesa (no gru)",
}

STATE_ORDER = [STATE_WORK, STATE_STANDBY, STATE_STOP_NIGHT, STATE_WAIT_NO_CRANE]

OPEN_METEO_ENSEMBLE_ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_DAYS_FIXED = 7  # sempre 7 giorni reali da Open-Meteo


# ==============================
# UTILITIES
# ==============================
def to_time(hhmm: str):
    return pd.to_datetime(hhmm, format="%H:%M").time()


def is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5


def in_work_shift(ts: pd.Timestamp, shift_start: str, shift_end: str) -> bool:
    t = ts.time()
    return (t >= to_time(shift_start)) and (t < to_time(shift_end))


def shift_end_timestamp(day_or_ts: pd.Timestamp, shift_end: str) -> pd.Timestamp:
    return pd.Timestamp(day_or_ts.date()) + pd.to_timedelta(f"{shift_end}:00")


def safe_percentile(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.percentile(a, q))


def safe_float(v, default=None):
    try:
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return default
            return float(s.replace(",", "."))
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_bool(v, default=True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y", "si", "sì"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return default


def format_time_estimate(seconds: float) -> str:
    if seconds is None or seconds <= 0:
        return "n.d."
    if seconds < 1:
        return "< 1 secondo"
    if seconds < 60:
        return f"~{seconds:.0f} secondi"
    return f"~{seconds/60:.1f} minuti"


def heuristic_estimate_seconds(n_members: int, n_days: int, horizon_hours: int, total_work_h: float) -> float:
    """
    Stima euristica molto prudente: la nuova simulazione è vettorizzata sui membri ma
    mantiene un loop sui D0 e sul tempo, oltre che sugli step.
    """
    if n_members <= 0 or n_days <= 0:
        return 0.0
    return max(0.5, n_days * (horizon_hours * 0.0012 + total_work_h * 0.015 + n_members * 0.001))


def shift_len_hours(shift_start: str, shift_end: str) -> float:
    t0 = pd.to_datetime(shift_start, format="%H:%M")
    t1 = pd.to_datetime(shift_end, format="%H:%M")
    return max(1.0, (t1 - t0).total_seconds() / 3600.0)


def average_timestamp(ts_array: np.ndarray) -> pd.Timestamp:
    ts_array = pd.to_datetime(ts_array, errors="coerce")
    ts_array = ts_array[~pd.isna(ts_array)]
    if len(ts_array) == 0:
        return pd.NaT
    nums = ts_array.view("int64")
    return pd.to_datetime(int(np.mean(nums)))


# ==============================
# POWER CURVE
# ==============================
def power_curve_mw(wind_ms: float, rated_mw: float = 2.0, cut_in: float = 3.0,
                   rated: float = 12.0, cut_out: float = 25.0) -> float:
    if wind_ms is None or (isinstance(wind_ms, float) and np.isnan(wind_ms)):
        return 0.0
    w = float(wind_ms)
    if w < cut_in:
        return 0.0
    if w < rated:
        return rated_mw * ((w - cut_in) / (rated - cut_in)) ** 3
    if w < cut_out:
        return rated_mw
    return 0.0


def power_curve_mw_array(wind_ms_array, rated_mw: float = 2.0, cut_in: float = 3.0,
                         rated: float = 12.0, cut_out: float = 25.0):
    """
    Versione numpy vettorizzata della power curve.
    Accetta scalare, vettore o matrice e restituisce array della stessa shape.
    """
    w = np.asarray(wind_ms_array, dtype=float)
    out = np.zeros_like(w, dtype=float)

    valid = np.isfinite(w)
    mask_ramp = valid & (w >= cut_in) & (w < rated)
    mask_rated = valid & (w >= rated) & (w < cut_out)

    out[mask_ramp] = rated_mw * ((w[mask_ramp] - cut_in) / (rated - cut_in)) ** 3
    out[mask_rated] = rated_mw
    return out


def plot_power_curve() -> go.Figure:
    wind = np.linspace(0, 30, 300)
    power = power_curve_mw_array(wind)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wind, y=power, mode="lines",
        line=dict(color="rgba(34,197,94,1)", width=3),
        name="Power Curve",
        hovertemplate="Vento: %{x:.1f} m/s<br>Potenza: %{y:.2f} MW<extra></extra>",
    ))
    fig.update_layout(
        title="Power Curve (2 MW)",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="Power [MW]",
        template="plotly_white",
        height=320,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ==============================
# PRICE SERIES (mock)
# ==============================
def generate_price_series(ts: pd.Series, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hour = ts.dt.hour.values
    dow = ts.dt.dayofweek.values
    base = (
        70
        + 10 * np.sin((hour - 6) / 24 * 2 * np.pi)
        + 25 * np.exp(-((hour - 20) ** 2) / (2 * 2.8 ** 2))
        + 10 * np.exp(-((hour - 9) ** 2) / (2 * 2.8 ** 2))
    )
    price = np.clip(base + np.where(dow >= 5, -7, 0) + rng.normal(0, 3.0, len(ts)), 60, 120)
    return np.round(price, 2)


# ==============================
# OPEN-METEO FETCH
# ==============================
def _extract_member_cols(hourly: dict, base_name: str) -> List[str]:

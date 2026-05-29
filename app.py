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

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="WTG Main Component - Cost Minimizer (Stochastic)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# UTILITIES
# -----------------------------
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
    if n_members <= 0 or n_days <= 0:
        return 0.0
    avg_h = max(1.0, horizon_hours / 2.0)
    return n_members * n_days * (avg_h + 2.0 * total_work_h) / 45000.0

def shift_len_hours(shift_start: str, shift_end: str) -> float:
    t0 = pd.to_datetime(shift_start, format="%H:%M")
    t1 = pd.to_datetime(shift_end, format="%H:%M")
    return max(1.0, (t1 - t0).total_seconds() / 3600.0)

# -----------------------------
# POWER CURVE (2 MW reference)
# -----------------------------
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

def plot_power_curve() -> go.Figure:
    wind = np.linspace(0, 30, 300)
    power = np.array([power_curve_mw(w) for w in wind], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wind, y=power, mode="lines",
        line=dict(color="rgba(34,197,94,1)", width=3),
        name="Power Curve",
        hovertemplate="Vento: %{x:.1f} m/s<br>Potenza: %{y:.2f} MW<extra></extra>",
    ))
    fig.update_layout(
        title="Power Curve (2 MW)", xaxis_title="Wind speed [m/s]", yaxis_title="Power [MW]",
        template="plotly_white", height=320, hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

# -----------------------------
# PRICES (mock)
# -----------------------------
def generate_price_series(ts: pd.Series, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hour = ts.dt.hour.values
    dow = ts.dt.dayofweek.values
    base = (70 + 10 * np.sin((hour - 6) / 24 * 2 * np.pi)
            + 25 * np.exp(-((hour - 20) ** 2) / (2 * 2.8 ** 2))
            + 10 * np.exp(-((hour - 9) ** 2) / (2 * 2.8 ** 2)))
    price = np.clip(base + np.where(dow >= 5, -7, 0) + rng.normal(0, 3.0, len(ts)), 60, 120)
    return np.round(price, 2)

# -----------------------------
# OPEN-METEO FETCH (always 7 days)
# -----------------------------
OPEN_METEO_ENSEMBLE_ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_DAYS_FIXED = 7   # ← hardcoded: always request 7 days from OpenMeteo

def _extract_member_cols(hourly: dict, base_name: str) -> List[str]:
    """
    Match keys like:
      wind_speed_80m_member01   (GFS/ICON/ECMWF style, zero-padded, no sep)
      wind_speed_80m_member_01  (with underscore sep)
      wind_speed_80m_member1    (no padding)
    Returns list sorted by member index (numeric, ignoring leading zeros).
    """
    patt = re.compile(rf"^{re.escape(base_name)}_?member[\s_]?(\d+)$")
    found = []
    for k in hourly.keys():
        m = patt.match(k)
        if m:
            found.append((int(m.group(1)), k))
    found.sort(key=lambda x: x[0])
    return [k for _, k in found]

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_open_meteo_ensemble(latitude: float, longitude: float, model: str,
                               include_gusts: bool, timezone: str = "auto") -> pd.DataFrame:
    """
    Fetch 7-day ensemble from Open-Meteo.
    Variables requested: wind_speed_80m (+ wind_gusts_10m if enabled).
    Open-Meteo returns one key per ensemble member, e.g.:
      wind_speed_80m_member01 ... wind_speed_80m_member31  (GFS: 31 members)
      wind_speed_80m_member01 ... wind_speed_80m_member40  (ICON: 40 members)
      wind_speed_80m_member00 ... wind_speed_80m_member50  (ECMWF IFS04: 51 members)
    Note: wind_gusts_10m is available for GFS and ICON but NOT for ecmwf_ifs04.
    """
    hourly_vars = ["wind_speed_80m"]
    if include_gusts:
        hourly_vars.append("wind_gusts_10m")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "models": model,
        "forecast_days": FORECAST_DAYS_FIXED,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
        "wind_speed_unit": "ms",   # force m/s (default is km/h for some endpoints!)
    }

    r = requests.get(OPEN_METEO_ENSEMBLE_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    if "hourly" not in js or "time" not in js["hourly"]:
        raise ValueError("Risposta Open-Meteo non contiene 'hourly.time'.")

    hourly = js["hourly"]
    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})

    # --- Wind speed members ---
    wind_keys = _extract_member_cols(hourly, "wind_speed_80m")
    if not wind_keys:
        # Fallback: try alternate roots in case model uses different naming
        candidates = sorted([k for k in hourly if "wind_speed" in k and "member" in k])
        if candidates:
            root = re.sub(r"_?member.*$", "", candidates[0])
            wind_keys = _extract_member_cols(hourly, root)
    if not wind_keys:
        raise ValueError(
            f"Impossibile trovare colonne ensemble 'wind_speed_80m_member*' per il modello '{model}'.\n"
            f"Chiavi disponibili (prime 15): {list(hourly.keys())[:15]}"
        )

    for i, k in enumerate(wind_keys):
        df[f"wind_speed_80m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    # --- Gust members (optional, not available for all models) ---
    if include_gusts:
        gust_keys = _extract_member_cols(hourly, "wind_gusts_10m")
        if gust_keys:
            n = min(len(gust_keys), len(wind_keys))
            for i, k in enumerate(gust_keys[:n]):
                df[f"wind_gusts_10m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")
        else:
            # Model doesn't provide gusts (e.g. ecmwf_ifs04) - silently skip
            pass

    df["price_eur_mwh"] = generate_price_series(df["timestamp"])
    return df

# -----------------------------
# MOCK DATA
# -----------------------------
def generate_mock_open_meteo_ensemble(days: int = 7, n_members: int = 30,
                                       seed: int = 42, include_gusts: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.now().normalize().strftime("%Y-%m-%d %H:%M")
    idx = pd.date_range(start=start, periods=days * 24, freq="1h")
    df = pd.DataFrame({"timestamp": idx})
    df["price_eur_mwh"] = generate_price_series(df["timestamp"], seed=seed + 1)

    hour = df["timestamp"].dt.hour.values
    day_f = ((df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 86400).values
    di = np.floor(day_f).astype(int)

    w_base = (7.5 + 1.8 * np.sin(hour / 24 * 2 * np.pi)
              + 1.2 * np.sin(day_f / 7 * 2 * np.pi)
              + 0.6 * np.sin((hour - 14) / 24 * 4 * np.pi))
    sigma = np.clip(np.where(di < 3, 0.9, 0.9 + (di - 2) * (3.2 / max(1, days - 3))), 0.9, 4.0)

    for m in range(n_members):
        wind = np.clip(w_base + rng.normal(0, 0.4) + rng.normal(0, sigma, len(df)), 0, 30)
        df[f"wind_speed_80m_member_{m}"] = np.round(wind, 2)
        if include_gusts:
            gust = np.clip(wind + np.abs(rng.normal(3.5, 1.4, len(df))) + rng.normal(0, sigma * 0.25, len(df)), 0, 40)
            df[f"wind_gusts_10m_member_{m}"] = np.round(gust, 2)

    return df

def detect_members(df: pd.DataFrame) -> Tuple[List[str], Optional[List[str]]]:
    wc = sorted([c for c in df.columns if c.startswith("wind_speed_80m_member_")],
                key=lambda c: int(c.split("_")[-1]))
    gc = sorted([c for c in df.columns if c.startswith("wind_gusts_10m_member_")],
                key=lambda c: int(c.split("_")[-1]))
    gc = gc if gc and len(gc) == len(wc) else None
    return wc, gc

# -----------------------------
# MONTE CARLO EXTENSION
# -----------------------------
def extend_ensemble_monte_carlo(df: pd.DataFrame, wind_cols: List[str],
                                 gust_cols: Optional[List[str]], extension_days: int = 7) -> pd.DataFrame:
    if df.empty or not wind_cols:
        return pd.DataFrame()

    fe = df["timestamp"].max()
    ext_idx = pd.date_range(start=fe + pd.Timedelta(hours=1), periods=extension_days * 24, freq="1h")
    df_ext = pd.DataFrame({"timestamp": ext_idx})
    df_ext["price_eur_mwh"] = generate_price_series(df_ext["timestamp"], seed=99)

    rng = np.random.default_rng(12345)
    phi = 0.92  # AR(1) mean-reversion coefficient

    for m, wc in enumerate(wind_cols):
        series = df[wc].dropna().to_numpy(dtype=float)
        mu = float(np.mean(series)) if len(series) else 7.5
        sd = float(np.std(series)) if len(series) else 2.5
        sig = max(0.2, sd * math.sqrt(1 - phi ** 2))

        last = series[-1] if len(series) else mu
        vals = []
        for _ in range(len(df_ext)):
            v = mu + phi * (last - mu) + rng.normal(0, sig)
            v = float(np.clip(v, 0.0, 30.0))
            vals.append(v)
            last = v
        df_ext[wc] = np.round(vals, 2)

        if gust_cols and m < len(gust_cols):
            gc = gust_cols[m]
            gs = df[gc].dropna().to_numpy(dtype=float)
            if len(gs) == len(series):
                md = max(1.0, float(np.mean(gs - series)))
                sd_d = max(0.5, float(np.std(gs - series)))
            else:
                md, sd_d = 3.5, 1.0
            gust_vals = [float(np.clip(v + max(0.5, md + rng.normal(0, sd_d * 0.5)), 0.0, 40.0)) for v in vals]
            df_ext[gc] = np.round(gust_vals, 2)

    return df_ext

# -------------------------------------------------------
# COMPUTE MC MEAN FOR BASE 7-DAY PERIOD
# (used to overlay a single mean trace on OpenMeteo period)
# -------------------------------------------------------
def compute_mc_mean_on_base(df_base: pd.DataFrame, wind_cols: List[str],
                             gust_cols: Optional[List[str]]) -> pd.DataFrame:
    """
    Compute, for each hour in df_base, the AR(1) MC *expected mean* (deterministic).
    This is the per-member mean trajectory, then averaged across members.
    We use the same AR(1) process as extend_ensemble_monte_carlo but run it
    forward from each timestamp independently to get E[wind | last_observed].
    In practice, the simplest defensible approach: compute the per-member
    rolling forecast mean = mu + phi^k * (last - mu), averaged over members.
    We approximate: for each hour t in base, use the *mean* of all members
    as the 'MC expected value' (since MC converges to ensemble mean in AR(1)).
    """
    # For the base period the MC expected mean simply equals the ensemble mean
    # (AR(1) unconditional mean = process mean = mean of the ensemble).
    # We return the ensemble mean resampled to the MC process's unconditional mean
    # per timestamp — which for display purposes is just the ensemble mean.
    if df_base.empty or not wind_cols:
        return pd.DataFrame()
    wind_mat = df_base[wind_cols].to_numpy(dtype=float)
    mc_mean = np.mean(wind_mat, axis=1)
    return pd.DataFrame({"timestamp": df_base["timestamp"].values, "mc_mean_wind": mc_mean})

# -----------------------------
# PLOTS
# -----------------------------
def _band_traces(x, p10: np.ndarray, p90: np.ndarray, mean: np.ndarray,
                 fill_color: str, line_color: str, label_suffix: str,
                 show_band: bool = True) -> list:
    """
    Returns list of traces for a band plot.
    If show_band=False, only the mean line is returned (no filled band).
    Tooltips correctly expose P10, mean, and P90 via customdata.
    """
    # customdata shape: (N, 3) → [p10, mean, p90]
    cd = np.column_stack([p10, mean, p90])
    traces = []
    if show_band:
        # Upper bound (P90) — invisible line, used as fill reference
        traces.append(go.Scatter(
            x=x, y=p90,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        # Lower bound (P10) — fill to previous (P90)
        traces.append(go.Scatter(
            x=x, y=p10,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill_color,
            name=f"P10–P90 {label_suffix}",
            customdata=cd,
            hovertemplate=(
                "<b>%{x|%d/%m %H:%M}</b><br>"
                "P90: %{customdata[2]:.2f}<br>"
                "Media: %{customdata[1]:.2f}<br>"
                "P10: %{customdata[0]:.2f}"
                "<extra>" + label_suffix + "</extra>"
            ),
        ))
    # Mean line
    traces.append(go.Scatter(
        x=x, y=mean,
        mode="lines",
        line=dict(color=line_color, width=2.5),
        name=f"Media {label_suffix}",
        customdata=cd,
        hovertemplate=(
            "<b>%{x|%d/%m %H:%M}</b><br>"
            "Media: %{y:.2f}<br>"
            "P10: %{customdata[0]:.2f}<br>"
            "P90: %{customdata[2]:.2f}"
            "<extra>" + label_suffix + "</extra>"
        ),
    ))
    return traces


def plot_wind_speed_ensemble(
    df_base: pd.DataFrame, wind_cols: List[str],
    df_ext: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    v_label_base = "(OpenMeteo 7gg)"
    v_label_ext  = "(Monte Carlo)"

    # ── Base period: OpenMeteo ensemble bands + MC mean overlay ──
    if not df_base.empty and wind_cols:
        wm = df_base[wind_cols].to_numpy(dtype=float)
        p10 = np.percentile(wm, 10, axis=1)
        p90 = np.percentile(wm, 90, axis=1)
        mean = np.mean(wm, axis=1)
        x = df_base["timestamp"].to_numpy()
        for t in _band_traces(x, p10, p90, mean,
                               fill_color="rgba(14,165,233,0.18)",
                               line_color="rgba(14,165,233,1)",
                               label_suffix=v_label_base,
                               show_band=True):
            fig.add_trace(t)
        # MC mean overlay on base period (no band, dashed line)
        mc_df = compute_mc_mean_on_base(df_base, wind_cols, None)
        if not mc_df.empty:
            mc_cd = np.column_stack([mc_df["mc_mean_wind"].values,
                                     mc_df["mc_mean_wind"].values,
                                     mc_df["mc_mean_wind"].values])
            fig.add_trace(go.Scatter(
                x=mc_df["timestamp"].to_numpy(),
                y=mc_df["mc_mean_wind"].values,
                mode="lines",
                line=dict(color="rgba(249,115,22,0.75)", width=1.8, dash="dot"),
                name=f"Media MC {v_label_base}",
                customdata=mc_cd,
                hovertemplate=(
                    "<b>%{x|%d/%m %H:%M}</b><br>"
                    "Media MC: %{y:.2f} m/s"
                    "<extra>MC su base</extra>"
                ),
            ))

    # ── Extension period: MC bands + mean ──
    if df_ext is not None and not df_ext.empty and wind_cols:
        # connect last point of base to first of ext for visual continuity
        conn = (pd.concat([df_base.tail(1), df_ext], ignore_index=True)
                if not df_base.empty else df_ext)
        wm_ext = conn[wind_cols].to_numpy(dtype=float)
        p10_e = np.percentile(wm_ext, 10, axis=1)
        p90_e = np.percentile(wm_ext, 90, axis=1)
        mean_e = np.mean(wm_ext, axis=1)
        x_e = conn["timestamp"].to_numpy()
        for t in _band_traces(x_e, p10_e, p90_e, mean_e,
                               fill_color="rgba(249,115,22,0.18)",
                               line_color="rgba(249,115,22,1)",
                               label_suffix=v_label_ext,
                               show_band=True):
            fig.add_trace(t)

    fig.update_layout(
        title="Velocità vento prevista [m/s]",
        xaxis_title="Tempo", yaxis_title="m/s",
        template="plotly_white", height=340,
        hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_expected_production(
    df_base: pd.DataFrame, wind_cols: List[str],
    df_ext: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    v_pw = np.vectorize(power_curve_mw)
    v_label_base = "(OpenMeteo 7gg)"
    v_label_ext  = "(Monte Carlo)"

    # ── Base period: OpenMeteo ──
    if not df_base.empty and wind_cols:
        wm = df_base[wind_cols].to_numpy(dtype=float)
        pm = v_pw(wm)
        p10 = np.percentile(pm, 10, axis=1)
        p90 = np.percentile(pm, 90, axis=1)
        mean = np.mean(pm, axis=1)
        x = df_base["timestamp"].to_numpy()
        for t in _band_traces(x, p10, p90, mean,
                               fill_color="rgba(59,130,246,0.18)",
                               line_color="rgba(59,130,246,1)",
                               label_suffix=v_label_base,
                               show_band=True):
            fig.add_trace(t)
        # MC mean overlay
        mc_df = compute_mc_mean_on_base(df_base, wind_cols, None)
        if not mc_df.empty:
            mc_pwr = np.array([power_curve_mw(w) for w in mc_df["mc_mean_wind"].values])
            mc_cd = np.column_stack([mc_pwr, mc_pwr, mc_pwr])
            fig.add_trace(go.Scatter(
                x=mc_df["timestamp"].to_numpy(), y=mc_pwr,
                mode="lines",
                line=dict(color="rgba(168,85,247,0.75)", width=1.8, dash="dot"),
                name=f"Media MC {v_label_base}",
                customdata=mc_cd,
                hovertemplate=(
                    "<b>%{x|%d/%m %H:%M}</b><br>"
                    "Media MC: %{y:.2f} MW"
                    "<extra>MC su base</extra>"
                ),
            ))

    # ── Extension period: MC bands ──
    if df_ext is not None and not df_ext.empty and wind_cols:
        conn = (pd.concat([df_base.tail(1), df_ext], ignore_index=True)
                if not df_base.empty else df_ext)
        wm_e = conn[wind_cols].to_numpy(dtype=float)
        pm_e = v_pw(wm_e)
        p10_e = np.percentile(pm_e, 10, axis=1)
        p90_e = np.percentile(pm_e, 90, axis=1)
        mean_e = np.mean(pm_e, axis=1)
        x_e = conn["timestamp"].to_numpy()
        for t in _band_traces(x_e, p10_e, p90_e, mean_e,
                               fill_color="rgba(168,85,247,0.18)",
                               line_color="rgba(168,85,247,1)",
                               label_suffix=v_label_ext,
                               show_band=True):
            fig.add_trace(t)

    fig.update_layout(
        title="Produzione prevista [MW]",
        xaxis_title="Tempo", yaxis_title="MW",
        template="plotly_white", height=340,
        hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# -----------------------------
# SIMULATION CORE (cost)
# -----------------------------
@dataclass
class CraneParams:
    mob_demob_eur: float
    op_cost_std_eur_h: float
    op_cost_fest_eur_h: float
    standby_cost_eur_h: float
    shift_start: str
    shift_end: str

@dataclass
class Step:
    name: str
    duration_h: float
    wind_thr: float
    gust_thr: Optional[float]
    min_seq_h: float
    requires_crane: bool


def check_min_window(df: pd.DataFrame, start_idx: int, wind_col: str,
                     gust_col: Optional[str], step: Step, params: CraneParams) -> bool:
    needed = int(math.ceil(max(0.0, step.min_seq_h)))
    if needed <= 1:
        return True
    ts0 = df.at[start_idx, "timestamp"]
    se_ts = shift_end_timestamp(ts0, params.shift_end)
    for k in range(needed):
        idx = start_idx + k
        if idx >= len(df):
            return False
        ts = df.at[idx, "timestamp"]
        if ts >= se_ts or not in_work_shift(ts, params.shift_start, params.shift_end):
            return False
        w = df.at[idx, wind_col]
        if pd.isna(w):
            return False
        ok = float(w) < float(step.wind_thr)
        if gust_col is not None and step.gust_thr is not None:
            g = df.at[idx, gust_col]
            if pd.isna(g):
                return False
            ok = ok and (float(g) < float(step.gust_thr))
        if not ok:
            return False
    return True


def op_cost_for_hour(ts: pd.Timestamp, params: CraneParams) -> float:
    return params.op_cost_fest_eur_h if is_weekend(ts) else params.op_cost_std_eur_h


def count_remaining_work_hours(df: pd.DataFrame, d0_ts: pd.Timestamp, params: CraneParams) -> int:
    mask = df["timestamp"] >= d0_ts
    return int(np.sum(df.loc[mask, "timestamp"]
               .apply(lambda x: in_work_shift(x, params.shift_start, params.shift_end))
               .to_numpy(dtype=bool)))


def simulate_single_start_day_cost(
    df: pd.DataFrame, start_day: pd.Timestamp,
    wind_cols: List[str], gust_cols: Optional[List[str]],
    steps: List[Step], params: CraneParams, rated_mw: float = 2.0,
) -> Dict:
    df = df.sort_values("timestamp").reset_index(drop=True)
    d0_ts = pd.Timestamp(start_day.date())
    d0_idx = int(df["timestamp"].searchsorted(d0_ts))
    if d0_idx >= len(df):
        return {"status": "out_of_range"}
    d0_ts = df.at[d0_idx, "timestamp"]

    mob = params.mob_demob_eur if any(s.requires_crane for s in steps) else 0.0
    member_rows = []
    member_logs = []

    for m, wc in enumerate(wind_cols):
        gc = gust_cols[m] if gust_cols else None
        si = 0; rem = float(steps[0].duration_h); started = False
        crane_c = 0.0; prod_l = 0.0
        idx = d0_idx; last_ts = d0_ts
        logs = []

        while idx < len(df) and si < len(steps):
            ts = df.at[idx, "timestamp"]
            last_ts = ts
            step = steps[si]

            w = df.at[idx, wc]
            g = df.at[idx, gc] if gc else np.nan
            p_mw = power_curve_mw(w, rated_mw=rated_mw)
            price = float(df.at[idx, "price_eur_mwh"])
            loss_eur = p_mw * price
            prod_l += loss_eur

            cp = any(s.requires_crane for s in steps[si:])
            c = 0.0

            if not in_work_shift(ts, params.shift_start, params.shift_end):
                state = "Stop Notte"
            else:
                ok = float(w) < float(step.wind_thr) if not pd.isna(w) else False
                if gc and step.gust_thr is not None and not pd.isna(g):
                    ok = ok and float(g) < float(step.gust_thr)
                win = check_min_window(df, idx, wc, gc, step, params) if not started else True

                if ok and win:
                    state = "Lavoro"
                    rem -= min(1.0, rem); started = True
                    if cp:
                        c = op_cost_for_hour(ts, params)
                else:
                    state = "Standby" if cp else "Attesa (no gru)"
                    if cp:
                        c = params.standby_cost_eur_h
                crane_c += c

                if rem <= 1e-9:
                    si += 1
                    if si < len(steps):
                        rem = float(steps[si].duration_h); started = False

            logs.append({
                "timestamp": ts, "state": state, "crane_cost_eur": c,
                "prod_loss_eur": loss_eur, "step_name": step.name, "member": m,
            })
            idx += 1

        member_rows.append({
            "member": m,
            "total_cost_eur": mob + crane_c + prod_l,
            "mob_demob_eur": mob, "crane_cost_eur": crane_c, "lost_revenue_eur": prod_l,
            "partial": si < len(steps), "completion_ts": last_ts,
        })
        member_logs.append(pd.DataFrame(logs))

    return {
        "status": "ok", "start_day": d0_ts,
        "member_results": pd.DataFrame(member_rows),
        "member_logs": member_logs,
    }


def compute_daily_summary_cost(all_sims: Dict, structural_infeasible: Dict) -> pd.DataFrame:
    rows = []
    for d0, sim in all_sims.items():
        if sim.get("status") != "ok":
            continue
        mr = sim["member_results"]
        costs = mr["total_cost_eur"].to_numpy(dtype=float)
        n = len(mr)
        success = int(np.sum(~mr["partial"].to_numpy(dtype=bool)))
        prob_ok = (success / n * 100.0) if n else 0.0
        p10 = safe_percentile(costs, 10)
        p90 = safe_percentile(costs, 90)
        mean = float(np.nanmean(costs[np.isfinite(costs)])) if np.any(np.isfinite(costs)) else np.nan
        spread = (p90 - p10) if np.isfinite(p10) and np.isfinite(p90) else np.nan
        rows.append({
            "Giorno Inizio (D0)": pd.Timestamp(d0).date(),
            "Probabilità Successo (%)": prob_ok,
            "Costo P10 €": p10, "Costo Medio €": mean, "Costo P90 €": p90,
            "Spread (P90-P10) €": spread,
            "Strutturalmente impossibile": bool(structural_infeasible.get(pd.Timestamp(d0), False)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Giorno Inizio (D0)").reset_index(drop=True)


def choose_optimal_day_cost(summary: pd.DataFrame, last_possible_start: Optional[pd.Timestamp],
                             risk_aversion: float = 0.7) -> Tuple[Optional[pd.Timestamp], pd.DataFrame, str]:
    s = summary.copy()
    if s.empty:
        return None, s, "Nessun dato"
    mean = s["Costo Medio €"].to_numpy(dtype=float)
    spread = np.nan_to_num(s["Spread (P90-P10) €"].to_numpy(dtype=float), nan=0.0)
    score = mean + float(risk_aversion) * spread
    s["Score (min meglio)"] = score
    if last_possible_start is None:
        return None, s, "Nessun D0 completabile nelle ore turno disponibili"
    cands = s[pd.to_datetime(s["Giorno Inizio (D0)"]) <= pd.Timestamp(last_possible_start.date())].copy()
    if cands.empty:
        return None, s, "Tutti i D0 candidabili sono oltre l'ultimo giorno utile"
    if np.nanmax(cands["Probabilità Successo (%)"].to_numpy(dtype=float)) <= 0.0:
        return None, s, "Probabilità di completamento = 0% per tutti i D0"
    best_idx = int(np.nanargmin(cands["Score (min meglio)"].to_numpy(dtype=float)))
    best_day = pd.Timestamp(cands.iloc[best_idx]["Giorno Inizio (D0)"])
    return best_day, s, ""


def plot_cost_candles(summary_for_plot: pd.DataFrame, best_day: Optional[pd.Timestamp] = None) -> go.Figure:
    if summary_for_plot.empty:
        return go.Figure()
    dfp = summary_for_plot.copy().sort_values("Giorno Inizio (D0)")
    x = dfp["Giorno Inizio (D0)"].astype(str)
    p10 = dfp["Costo P10 €"].to_numpy(dtype=float)
    p90 = dfp["Costo P90 €"].to_numpy(dtype=float)
    mean = dfp["Costo Medio €"].to_numpy(dtype=float)
    body = p90 - p10

    fig = go.Figure()
    # Bar body P10→P90: pass p90 as customdata so tooltip can show it
    fig.add_trace(go.Bar(
        x=x, y=body, base=p10,
        marker=dict(color="rgba(99,102,241,0.35)"),
        name="Intervallo P10–P90",
        customdata=p90,
        hovertemplate=(
            "D0: %{x}<br>"
            "P10: %{base:,.0f} €<br>"
            "P90: %{customdata:,.0f} €"
            "<extra>P10–P90</extra>"
        ),
    ))
    # Mean line with full info in tooltip
    cd_mean = np.column_stack([p10, mean, p90])
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines+markers",
        line=dict(color="rgba(99,102,241,1)", width=2),
        marker=dict(size=6),
        name="Costo medio",
        customdata=cd_mean,
        hovertemplate=(
            "D0: %{x}<br>"
            "P10: %{customdata[0]:,.0f} €<br>"
            "Media: %{y:,.0f} €<br>"
            "P90: %{customdata[2]:,.0f} €"
            "<extra>Media</extra>"
        ),
    ))
    # Highlight best day
    if best_day is not None:
        bd_str = str(best_day.date())
        bd_mask = dfp["Giorno Inizio (D0)"].astype(str) == bd_str
        if bd_mask.any():
            bd_mean = mean[bd_mask.to_numpy()][0]
            fig.add_trace(go.Scatter(
                x=[bd_str], y=[bd_mean], mode="markers",
                marker=dict(size=14, color="rgba(239,68,68,1)", symbol="star"),
                name="⭐ D0 ottimale",
                hovertemplate=f"D0 OTTIMALE: {bd_str}<br>Costo medio: {bd_mean:,.0f} €<extra></extra>",
            ))
    fig.update_layout(
        title="Costo totale vs D0 (candela P10–P90)",
        xaxis_title="Giorno di inizio D0", yaxis_title="Costo Totale (€)",
        template="plotly_white", height=430, hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10), barmode="overlay",
    )
    return fig


def aggregate_gantt(member_logs: List[pd.DataFrame]) -> pd.DataFrame:
    if not member_logs:
        return pd.DataFrame()
    all_logs = pd.concat(member_logs, ignore_index=True)
    n = all_logs["member"].nunique()
    pivot = all_logs.groupby(["timestamp", "state"])["member"].nunique().unstack(fill_value=0).sort_index()
    for col in ["Lavoro", "Standby", "Stop Notte", "Attesa (no gru)"]:
        if col not in pivot.columns:
            pivot[col] = 0
    return (pivot / max(1, n)).reset_index()


def plot_gantt_fraction(frac: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, color in [
        ("Lavoro",           "rgba(34,197,94,0.85)"),
        ("Standby",          "rgba(245,158,11,0.85)"),
        ("Attesa (no gru)",  "rgba(59,130,246,0.30)"),
        ("Stop Notte",       "rgba(148,163,184,0.85)"),
    ]:
        if name in frac.columns:
            fig.add_trace(go.Bar(
                x=frac["timestamp"], y=frac[name], name=name,
                marker_color=color,
                hovertemplate=f"<b>%{{x|%d/%m %H:%M}}</b><br>{name}: %{{y:.0%}}<extra></extra>",
            ))
    fig.update_layout(
        barmode="stack", title="Gantt medio (quota scenari per stato)",
        template="plotly_white", height=380, hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

# ==============================
# UI
# ==============================
st.title("WTG Main Component – Cost Minimizer (Stochastic)")

with st.sidebar:
    st.header("A) Parametri economici & operativi")
    mob_demob  = st.number_input("Mob/Demob [€]",                    value=45000.0, step=1000.0)
    op_std     = st.number_input("Costo operativo standard [€/h]",   value=1200.0,  step=50.0)
    op_fest    = st.number_input("Costo operativo weekend [€/h]",    value=1500.0,  step=50.0)
    standby    = st.number_input("Costo standby [€/h]",              value=650.0,   step=25.0)
    shift_start = st.text_input("Inizio turno", value="07:00")
    shift_end   = st.text_input("Fine turno",   value="18:00")

    st.divider()
    st.header("B) Open‑Meteo ensemble")
    st.caption(f"Forecast richiesto: **{FORECAST_DAYS_FIXED} giorni** (fisso). "
               "Oltre l'orizzonte reale disponibile viene generata un'estensione Monte Carlo.")
    latitude     = st.number_input("Latitudine",  value=41.5, format="%.6f")
    longitude    = st.number_input("Longitudine", value=15.2, format="%.6f")
    model        = st.selectbox("Modello", options=["gfs_seamless", "icon_seamless", "ecmwf_ifs04"])
    include_gusts = st.toggle("Usa raffiche (wind_gusts_10m)",    value=True,
                              help="ecmwf_ifs04 non fornisce raffiche; verrà ignorato se non disponibile.")

    st.divider()
    st.header("C) Campionamento")
    use_all_members  = st.toggle("Usa tutti i membri", value=False)
    n_members_input  = st.number_input("Membri da usare", min_value=1, value=10,
                                        disabled=use_all_members)

    st.divider()
    st.header("D) Ottimizzazione")
    earliest_day  = st.date_input("Data minima D0", value=pd.Timestamp.now().date())
    risk_aversion = st.slider("Risk aversion (penalità spread)", 0.0, 3.0, 0.7, 0.1)

    st.divider()
    st.header("E) Debug")
    use_mock = st.toggle("Mock Data (debug)", value=False)

# ── Validate shift ──
try:
    if to_time(shift_start) >= to_time(shift_end):
        st.error("Inizio turno deve essere precedente alla fine.")
        st.stop()
except Exception:
    st.error("Formato turno non valido (usa HH:MM).")
    st.stop()

params = CraneParams(
    mob_demob_eur=float(mob_demob), op_cost_std_eur_h=float(op_std),
    op_cost_fest_eur_h=float(op_fest), standby_cost_eur_h=float(standby),
    shift_start=shift_start, shift_end=shift_end,
)

# ── Steps table ──
st.subheader("Pianificazione Attività (step)")
default_steps = pd.DataFrame({
    "Step": ["Step 1", "Step 2", "Step 3"],
    "Durata [h]": [8.0, 8.0, 8.0],
    "Wind Threshold [m/s]": [8.0, 8.0, 8.0],
    "Gust Threshold [m/s] (opzionale)": [np.nan, np.nan, np.nan],
    "Finestra minima consecutiva [h]": [3.0, 3.0, 3.0],
    "Richiede Gru": [True, True, True],
})
steps_df = st.data_editor(default_steps, num_rows="dynamic", use_container_width=True, hide_index=True)

steps: List[Step] = []
for _, r in steps_df.iterrows():
    name = str(r.get("Step", "")).strip()
    dur  = safe_float(r.get("Durata [h]"))
    wt   = safe_float(r.get("Wind Threshold [m/s]"))
    minw = safe_float(r.get("Finestra minima consecutiva [h]"), default=0.0)
    gt_r = r.get("Gust Threshold [m/s] (opzionale)")
    gt   = None if (gt_r is None or (isinstance(gt_r, float) and np.isnan(gt_r))) else safe_float(gt_r)
    req  = safe_bool(r.get("Richiede Gru"), default=True)
    if name == "" and dur is None and wt is None:
        continue
    if not dur or dur <= 0 or not wt or wt <= 0:
        continue
    steps.append(Step(name=name or "Step", duration_h=float(dur), wind_thr=float(wt),
                      gust_thr=gt, min_seq_h=float(minw or 0), requires_crane=req))

if not steps:
    st.error("Inserisci almeno uno step valido (Durata > 0 e Wind Threshold > 0).")
    st.stop()

required_work_h = float(sum(s.duration_h for s in steps))

# ── Load meteo (always 7 days) ──
with st.spinner("Caricamento dati meteo..."):
    if use_mock:
        df = generate_mock_open_meteo_ensemble(days=FORECAST_DAYS_FIXED, n_members=30, include_gusts=include_gusts)
        st.info(f"Mock Data: {FORECAST_DAYS_FIXED} giorni, 30 membri, raffiche={'sì' if include_gusts else 'no'}.")
    else:
        try:
            df = fetch_open_meteo_ensemble(latitude, longitude, model, include_gusts)
            st.success("Forecast Open‑Meteo caricato.")
        except Exception as e:
            st.error(f"Errore API Open‑Meteo: {e}")
            st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

forecast_start = df["timestamp"].min()
forecast_end   = df["timestamp"].max()
actual_days    = int((forecast_end.normalize() - forecast_start.normalize()).days) + 1

if actual_days < FORECAST_DAYS_FIXED and not use_mock:
    st.warning(
        f"⚠️ Il modello ha restituito {actual_days} giorni di forecast "
        f"(attesi: {FORECAST_DAYS_FIXED}). L'estensione MC coprirà il gap."
    )
st.caption(
    f"OpenMeteo: {forecast_start.strftime('%d/%m/%Y %H:%M')} → "
    f"{forecast_end.strftime('%d/%m/%Y %H:%M')} ({actual_days} gg reali)"
)

wind_cols_all, gust_cols_all = detect_members(df)
if not wind_cols_all:
    st.error("Nessun membro ensemble trovato nel dataset. Controlla il modello selezionato.")
    st.stop()

n_avail = len(wind_cols_all)
wind_cols_use = wind_cols_all if use_all_members else wind_cols_all[:min(n_avail, int(n_members_input))]
gust_cols_use = gust_cols_all[:len(wind_cols_use)] if gust_cols_all else None

if not use_all_members and int(n_members_input) > n_avail:
    st.warning(f"Richiesti {n_members_input} membri, disponibili {n_avail}. Uso tutti.")

horizon_start = max(forecast_start.normalize(), pd.Timestamp(earliest_day))
all_days = pd.date_range(start=horizon_start.normalize(),
                          end=(forecast_end + pd.Timedelta(days=7)).normalize(),
                          freq="1D").to_list()
if not all_days:
    st.error("Nessun giorno D0 disponibile.")
    st.stop()

# ── Session state ──
for k in ["sims_cost", "sims_hash_cost", "df_extension"]:
    if k not in st.session_state:
        st.session_state[k] = None

cur_hash = hashlib.md5(json.dumps({
    "m": len(wind_cols_use),
    "days": [str(x.date()) for x in all_days],
    "steps": [(s.name, s.duration_h, s.wind_thr, s.gust_thr, s.min_seq_h, s.requires_crane) for s in steps],
    "params": [mob_demob, op_std, op_fest, standby, shift_start, shift_end],
    "ra": risk_aversion, "hs": str(horizon_start), "fe": str(forecast_end),
}, sort_keys=True).encode()).hexdigest()

if st.session_state["sims_cost"] is not None and st.session_state["sims_hash_cost"] != cur_hash:
    st.warning("⚠️ Parametri cambiati dall'ultima esecuzione. Premi il pulsante per aggiornare.")

# ── Determine active df (base + extension if available) ──
df_ext_stored = st.session_state["df_extension"]
if df_ext_stored is not None and st.session_state["sims_hash_cost"] == cur_hash:
    df_active = pd.concat([df, df_ext_stored], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df_ext_to_plot = df_ext_stored
else:
    df_active = df
    df_ext_to_plot = None

# ── Context plots ──
st.subheader("Contesto meteo & produzione")
df_view = df[(df["timestamp"] >= horizon_start) & (df["timestamp"] <= forecast_end)].copy()

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_power_curve(), use_container_width=True)
with c2:
    st.plotly_chart(plot_wind_speed_ensemble(df_view, wind_cols_use, df_ext_to_plot), use_container_width=True)
st.plotly_chart(plot_expected_production(df_view, wind_cols_use, df_ext_to_plot), use_container_width=True)

# ── Feasibility map ──
structural_infeasible  = {}
available_work_h_map   = {}
for d0 in all_days:
    d0_ts = pd.Timestamp(d0.date())
    idx0 = int(df_active["timestamp"].searchsorted(d0_ts))
    if idx0 >= len(df_active):
        structural_infeasible[d0_ts] = True
        available_work_h_map[d0_ts] = 0
        continue
    avail = count_remaining_work_hours(df_active, df_active.at[idx0, "timestamp"], params)
    available_work_h_map[d0_ts] = avail
    structural_infeasible[d0_ts] = avail < required_work_h

feasible = [pd.Timestamp(d) for d in all_days if not structural_infeasible.get(pd.Timestamp(d), False)]
last_possible_start = feasible[-1] if feasible else None
st.caption(f"Ultimo D0 possibile (ore turno sufficienti): "
           f"{last_possible_start.date() if last_possible_start else 'nessuno'}")

# ── Simulation controls ──
st.subheader("Simulazione stocastica (Costi totali)")
horizon_hours = int((forecast_end + pd.Timedelta(days=7) - horizon_start).total_seconds() / 3600) + 1
est_sec = heuristic_estimate_seconds(len(wind_cols_use), len(all_days), horizon_hours, required_work_h)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Giorni D0 simulati", len(all_days))
m2.metric("Membri ensemble usati", len(wind_cols_use))
m3.metric("Orizzonte totale (ore)", horizon_hours)
m4.metric("Simulazioni totali", len(all_days) * len(wind_cols_use))

c_btn, c_est = st.columns([1, 2])
with c_btn:
    run_clicked = st.button("▶ Esegui simulazione", type="primary", use_container_width=True)
with c_est:
    st.info(f"⏱️ Tempo stimato: **{format_time_estimate(est_sec)}** — "
            f"{len(wind_cols_use)} membri × {len(all_days)} giorni")

# ── Run simulation ──
if run_clicked:
    with st.spinner("Generazione estensione Monte Carlo..."):
        df_ext = extend_ensemble_monte_carlo(df, wind_cols_all, gust_cols_all, extension_days=7)
        st.session_state["df_extension"] = df_ext

    df_sim = pd.concat([df, df_ext], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    sims = {}
    prog = st.progress(0.0, text="Avvio simulazione...")
    t0_run = time.perf_counter()
    for i, d0 in enumerate(all_days):
        sims[pd.Timestamp(d0)] = simulate_single_start_day_cost(
            df=df_sim, start_day=pd.Timestamp(d0),
            wind_cols=wind_cols_use, gust_cols=gust_cols_use,
            steps=steps, params=params,
        )
        elapsed = time.perf_counter() - t0_run
        rem = (elapsed / (i + 1)) * (len(all_days) - i - 1)
        prog.progress((i + 1) / len(all_days),
                      text=f"D0 {i+1}/{len(all_days)} — rimanente: {format_time_estimate(rem)}")

    prog.empty()
    st.session_state["sims_cost"] = sims
    st.session_state["sims_hash_cost"] = cur_hash
    st.success(f"✅ Completata in {format_time_estimate(time.perf_counter() - t0_run)}.")
    st.rerun()

# ── Results ──
sims = st.session_state["sims_cost"]
if sims is None:
    st.info("👆 Configura i parametri e premi **Esegui simulazione** per avviare l'analisi.")
    st.stop()

summary_df = compute_daily_summary_cost(sims, structural_infeasible)
best_day, summary_scored, err_msg = choose_optimal_day_cost(summary_df, last_possible_start, risk_aversion)

if err_msg:
    st.error(f"❌ {err_msg}")

# KPI
if best_day is not None and not summary_scored.empty:
    br = summary_scored[summary_scored["Giorno Inizio (D0)"] == best_day.date()].iloc[0]
    kc = st.columns(4)
    kc[0].metric("🏆 Miglior D0", str(best_day.date()))
    kc[1].metric("Costo Medio Atteso", f"{br['Costo Medio €']:,.0f} €")
    kc[2].metric("Probabilità Successo", f"{br['Probabilità Successo (%)']:.1f} %")
    kc[3].metric("Spread P90-P10", f"{br['Spread (P90-P10) €']:,.0f} €")

# Cost candles chart
plot_df = summary_scored[~summary_scored["Strutturalmente impossibile"]] if not summary_scored.empty else pd.DataFrame()
if not plot_df.empty:
    st.plotly_chart(plot_cost_candles(plot_df, best_day), use_container_width=True)

# Summary table
st.subheader("Tabella risultati")
disp_cols = ["Giorno Inizio (D0)", "Probabilità Successo (%)",
             "Costo P10 €", "Costo Medio €", "Costo P90 €", "Spread (P90-P10) €",
             "Strutturalmente impossibile"]
if not summary_scored.empty:
    st.dataframe(
        summary_scored[disp_cols].style.format({
            "Probabilità Successo (%)": "{:.1f}",
            "Costo P10 €": "{:,.0f}", "Costo Medio €": "{:,.0f}",
            "Costo P90 €": "{:,.0f}", "Spread (P90-P10) €": "{:,.0f}",
        }),
        use_container_width=True,
    )

# Detail per D0
st.subheader("Dettaglio D0 selezionato")
selected = st.selectbox("Seleziona D0", [d.date() for d in all_days], key="sel_d0_detail")
sel_ts = pd.Timestamp(selected)
sim_sel = sims.get(sel_ts)

infeasible_sel = bool(structural_infeasible.get(sel_ts, False))
avail_sel = available_work_h_map.get(sel_ts, 0)
st.caption(
    f"D0 {selected}: "
    f"{'⛔ STRUTTURALMENTE IMPOSSIBILE' if infeasible_sel else '✅ strutturalmente completabile'} "
    f"(ore turno disponibili: {avail_sel} vs richieste: {required_work_h:.0f})"
)

if sim_sel and sim_sel.get("status") == "ok":
    mr = sim_sel["member_results"]
    ca = mr["total_cost_eur"].to_numpy(dtype=float)
    succ = 100.0 * float(np.mean(~mr["partial"].to_numpy(dtype=bool)))
    dc = st.columns(4)
    dc[0].metric("Probabilità successo", f"{succ:.1f} %")
    dc[1].metric("Costo Medio", f"{np.mean(ca):,.0f} €")
    dc[2].metric("Costo P10", f"{safe_percentile(ca, 10):,.0f} €")
    dc[3].metric("Costo P90", f"{safe_percentile(ca, 90):,.0f} €")

    frac = aggregate_gantt(sim_sel["member_logs"])
    if not frac.empty:
        st.plotly_chart(plot_gantt_fraction(frac), use_container_width=True)
elif sim_sel and sim_sel.get("status") == "out_of_range":
    st.warning("D0 fuori dall'orizzonte del dataset.")
else:
    st.info("Nessuna simulazione disponibile per questo D0.")

with st.expander("ℹ️ Note tecniche", expanded=False):
    st.markdown(f"""
**Forecast OpenMeteo:** sempre {FORECAST_DAYS_FIXED} giorni (fisso).  
**Estensione Monte Carlo:** AR(1) mean-reverting per ulteriori 7 giorni dopo il forecast.  
Nei grafici meteo/produzione, sul periodo OpenMeteo compare la **banda P10–P90 ensemble** (azzurro/blu)
e, sovrapposta, solo la **linea media MC** (arancione tratteggiata, senza bande).  
Sul periodo MC compare la **banda P10–P90 MC** con relativa media (arancione/viola pieno).  

**Bug P90 tooltip:** tutti i grafici ora usano `customdata` per esporre correttamente P10, media e P90
nei tooltip (il vecchio `%{{y+base}}` di Plotly non funziona).  

**Unità vento:** la fetch aggiunge `wind_speed_unit=ms` per forzare m/s (alcuni endpoint
Open-Meteo restituivano km/h di default, causando valori ~3.6× superiori).  

**Avvio:** `streamlit run app.py`
""")

st.caption("✅ streamlit run app.py")

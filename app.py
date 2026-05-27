import math
import re
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
    page_title="WTG Main Component - Decision Making Tool",
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


def shift_start_timestamp(day: pd.Timestamp, shift_start: str) -> pd.Timestamp:
    return pd.Timestamp(day.date()) + pd.to_timedelta(f"{shift_start}:00")


def shift_end_timestamp(day: pd.Timestamp, shift_end: str) -> pd.Timestamp:
    return pd.Timestamp(day.date()) + pd.to_timedelta(f"{shift_end}:00")


def safe_percentile(a: np.ndarray, q: float) -> float:
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.percentile(a, q))


def clamp_date_to_forecast(d: pd.Timestamp, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp) -> pd.Timestamp:
    if d < forecast_start:
        return forecast_start.normalize()
    if d > forecast_end:
        return forecast_end.normalize()
    return d.normalize()


# -----------------------------
# POWER CURVE (2 MW reference)
# -----------------------------
def power_curve_mw(
    wind_ms: float,
    rated_mw: float = 2.0,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
) -> float:
    if wind_ms is None or (isinstance(wind_ms, float) and np.isnan(wind_ms)):
        return 0.0
    w = float(wind_ms)
    if w < cut_in:
        return 0.0
    if w < rated:
        x = (w - cut_in) / (rated - cut_in)
        return rated_mw * (x ** 3)
    if w < cut_out:
        return rated_mw
    return 0.0


def plot_power_curve(
    rated_mw: float = 2.0,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
) -> go.Figure:
    wind = np.linspace(0, 30, 300)
    power = [power_curve_mw(w, rated_mw=rated_mw, cut_in=cut_in, rated=rated, cut_out=cut_out) for w in wind]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wind,
            y=power,
            mode="lines",
            line=dict(color="rgba(34,197,94,1)", width=3),
            name="Power curve",
        )
    )
    fig.update_layout(
        title="Power Curve Turbina (2 MW)",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="Power [MW]",
        template="plotly_white",
        height=340,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


# -----------------------------
# PRICES (mock if not provided)
# -----------------------------
def generate_price_series(ts: pd.Series, seed: int = 7) -> np.ndarray:
    """
    Demo price curve 60-120 €/MWh with intraday shape.
    Used when you don't load/compute real prices elsewhere.
    """
    rng = np.random.default_rng(seed)
    hour = ts.dt.hour.values
    dow = ts.dt.dayofweek.values

    base = (
        70
        + 10 * np.sin((hour - 6) / 24 * 2 * np.pi)
        + 25 * np.exp(-((hour - 20) ** 2) / (2 * 2.8**2))
        + 10 * np.exp(-((hour - 9) ** 2) / (2 * 2.8**2))
    )
    weekend_factor = np.where(dow >= 5, -7, 0)
    noise = rng.normal(0, 3.0, size=len(ts))
    price = np.clip(base + weekend_factor + noise, 60, 120)
    return np.round(price, 2)


# -----------------------------
# OPEN-METEO: FETCH ENSEMBLE
# -----------------------------
OPEN_METEO_ENSEMBLE_ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _extract_member_cols(hourly: dict, base_name: str) -> List[str]:
    """
    Robustly detect keys like:
      - wind_speed_80m_member_0
      - wind_speed_80m_member0
      - wind_speed_80m_member 0
    Returns ordered list by member index.
    """
    keys = list(hourly.keys())
    patt = re.compile(rf"^{re.escape(base_name)}_?member[\s_]?(\d+)$")
    found = []
    for k in keys:
        m = patt.match(k)
        if m:
            found.append((int(m.group(1)), k))
    found.sort(key=lambda x: x[0])
    return [k for _, k in found]


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_open_meteo_ensemble(
    latitude: float,
    longitude: float,
    model: str,
    forecast_days: int,
    include_gusts: bool,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Fetch ensemble forecast from Open-Meteo.
    We request wind_speed_80m and (optional) wind_gusts_10m.
    Forecast length up to 16 days per docs. 【1-2f3670】
    """
    hourly_vars = ["wind_speed_80m"]
    if include_gusts:
        hourly_vars.append("wind_gusts_10m")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "models": model,
        "forecast_days": int(forecast_days),
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }

    r = requests.get(OPEN_METEO_ENSEMBLE_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    if "hourly" not in js or "time" not in js["hourly"]:
        raise ValueError("Risposta Open-Meteo non contiene il blocco 'hourly.time'.")

    hourly = js["hourly"]
    times = pd.to_datetime(hourly["time"])
    df = pd.DataFrame({"timestamp": times})

    wind_keys = _extract_member_cols(hourly, "wind_speed_80m")
    if not wind_keys:
        # fallback: some models might provide only mean/spread; in that case stop early
        raise ValueError("Non trovo membri ensemble 'wind_speed_80m_member*' nella risposta Open-Meteo.")

    for i, k in enumerate(wind_keys):
        df[f"wind_speed_80m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    gust_keys = _extract_member_cols(hourly, "wind_gusts_10m") if include_gusts else []
    if include_gusts and gust_keys and len(gust_keys) == len(wind_keys):
        for i, k in enumerate(gust_keys):
            df[f"wind_gusts_10m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    # Add demo prices if none (Open-Meteo doesn't provide energy prices)
    df["price_eur_mwh"] = generate_price_series(df["timestamp"])

    return df


# -----------------------------
# MOCK DATA (optional fallback)
# -----------------------------
def generate_mock_open_meteo_ensemble(
    start: str = None,
    days: int = 10,
    n_members: int = 10,
    seed: int = 42,
    include_gusts: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if start is None:
        # start at today's midnight
        start = pd.Timestamp.now().normalize().strftime("%Y-%m-%d %H:%M")

    idx = pd.date_range(start=pd.to_datetime(start), periods=int(days * 24), freq="1h")
    df = pd.DataFrame({"timestamp": idx})

    df["price_eur_mwh"] = generate_price_series(df["timestamp"], seed=seed + 1)

    hour = df["timestamp"].dt.hour.values
    day_index = ((df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 86400).values
    di = np.floor(day_index).astype(int)

    w_base = (
        7.5
        + 1.8 * np.sin((hour) / 24 * 2 * np.pi)
        + 1.2 * np.sin((day_index) / 7 * 2 * np.pi)
        + 0.6 * np.sin((hour - 14) / 24 * 4 * np.pi)
    )

    sigma = np.where(di < 3, 0.9, 0.9 + (di - 2) * (3.2 / max(1, (days - 3))))
    sigma = np.clip(sigma, 0.9, 4.0)

    for m in range(n_members):
        member_bias = rng.normal(0, 0.4)
        member_noise = rng.normal(0, sigma, size=len(df))
        wind = np.clip(w_base + member_bias + member_noise, 0, 30)
        df[f"wind_speed_80m_member_{m}"] = np.round(wind, 2)

        if include_gusts:
            gust_extra = np.abs(rng.normal(3.5, 1.4, size=len(df))) + rng.normal(0, sigma * 0.25, size=len(df))
            gust = np.clip(wind + gust_extra, 0, 40)
            df[f"wind_gusts_10m_member_{m}"] = np.round(gust, 2)

    return df


# -----------------------------
# INPUT PARSING
# -----------------------------
def detect_members(df: pd.DataFrame) -> Tuple[List[str], Optional[List[str]]]:
    wind_cols = [c for c in df.columns if c.startswith("wind_speed_80m_member_")]
    wind_cols = sorted(wind_cols, key=lambda c: int(c.split("_")[-1]))

    gust_cols = [c for c in df.columns if c.startswith("wind_gusts_10m_member_")]
    gust_cols = sorted(gust_cols, key=lambda c: int(c.split("_")[-1])) if gust_cols else None

    if gust_cols is not None and len(gust_cols) != len(wind_cols):
        gust_cols = None

    return wind_cols, gust_cols


# -----------------------------
# PLOTS: PRICES & EXPECTED PRODUCTION
# -----------------------------
def plot_prices(df_view: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_view["timestamp"],
            y=df_view["price_eur_mwh"],
            mode="lines",
            line=dict(color="rgba(245,158,11,1)", width=2),
            name="Prezzo energia",
        )
    )
    fig.update_layout(
        title="Prezzi orari energia [€/MWh]",
        xaxis_title="Tempo",
        yaxis_title="€/MWh",
        template="plotly_white",
        height=340,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    return fig


def plot_expected_production(df_view: pd.DataFrame, wind_cols: List[str]) -> go.Figure:
    wind_mat = df_view[wind_cols].to_numpy(dtype=float)  # (T, M)
    v_power = np.vectorize(lambda w: power_curve_mw(w))
    power_mat = v_power(wind_mat)

    p10 = np.percentile(power_mat, 10, axis=1)
    p90 = np.percentile(power_mat, 90, axis=1)
    mean = np.mean(power_mat, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view["timestamp"], y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=df_view["timestamp"],
            y=p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.18)",
            name="Incertezza (P10–P90)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_view["timestamp"],
            y=mean,
            mode="lines",
            line=dict(color="rgba(59,130,246,1)", width=2.8),
            name="Produzione prevista (media)",
        )
    )
    fig.update_layout(
        title="Produzione prevista oraria [MW] (media ensemble + banda P10–P90)",
        xaxis_title="Tempo",
        yaxis_title="MW",
        template="plotly_white",
        height=340,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    return fig


# -----------------------------
# SIMULATION CORE
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


def check_min_window(
    df: pd.DataFrame,
    start_idx: int,
    wind_col: str,
    gust_col: Optional[str],
    step: Step,
    params: CraneParams,
) -> bool:
    needed = int(math.ceil(max(0.0, step.min_seq_h)))
    if needed <= 1:
        return True

    ts0 = df.at[start_idx, "timestamp"]
    shift_end_ts = shift_end_timestamp(ts0, params.shift_end)

    for k in range(needed):
        idx = start_idx + k
        if idx >= len(df):
            return False
        ts = df.at[idx, "timestamp"]
        if ts >= shift_end_ts:
            return False
        if not in_work_shift(ts, params.shift_start, params.shift_end):
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


def optimistic_completion_timestamp(d0: pd.Timestamp, total_work_h: float, params: CraneParams) -> pd.Timestamp:
    """
    Calendar completion with PERFECT weather:
    - work progresses only during shift hours
    - no standby
    - start of activity is at D0 00:00, but first workable hour is shift_start
    """
    remaining = float(total_work_h)
    t = pd.Timestamp(d0.date())  # 00:00

    while remaining > 1e-9:
        if in_work_shift(t, params.shift_start, params.shift_end):
            remaining -= 1.0
        t += pd.Timedelta(hours=1)

    return t


def simulate_single_start_day(
    df: pd.DataFrame,
    start_day: pd.Timestamp,
    wind_cols: List[str],
    gust_cols: Optional[List[str]],
    steps: List[Step],
    params: CraneParams,
    rated_mw: float = 2.0,
) -> Dict:
    start_ts = pd.Timestamp(start_day.date())  # 00:00
    df = df.sort_values("timestamp").reset_index(drop=True)

    start_idx = int(df["timestamp"].searchsorted(start_ts))
    if start_idx >= len(df):
        return {"status": "out_of_range"}

    member_results = []
    member_logs = []

    for m, wind_col in enumerate(wind_cols):
        gust_col = gust_cols[m] if gust_cols is not None else None

        step_i = 0
        remaining = float(steps[0].duration_h) if steps else 0.0
        current_step_started = False

        crane_cost = 0.0
        prod_loss = 0.0

        logs = []
        idx = start_idx
        incomplete = False
        last_ts = start_ts

        while step_i < len(steps):
            if idx >= len(df):
                incomplete = True
                break

            ts = df.at[idx, "timestamp"]
            last_ts = ts

            step = steps[step_i]
            step_name_for_log = step.name

            w = df.at[idx, wind_col]
            g = df.at[idx, gust_col] if gust_col is not None else np.nan

            # production loss always H24
            p_mw = power_curve_mw(w, rated_mw=rated_mw)
            price = float(df.at[idx, "price_eur_mwh"]) if "price_eur_mwh" in df.columns else 0.0
            loss_eur = p_mw * price
            prod_loss += loss_eur

            if not in_work_shift(ts, params.shift_start, params.shift_end):
                state = "Stop Notte"
                c_cost = 0.0
            else:
                ok = float(w) < float(step.wind_thr)
                if gust_col is not None and step.gust_thr is not None:
                    ok = ok and (float(g) < float(step.gust_thr))

                if not current_step_started:
                    window_ok = check_min_window(df, idx, wind_col, gust_col, step, params)
                    if window_ok and ok:
                        state = "Lavoro"
                        c_cost = op_cost_for_hour(ts, params)
                        work = min(1.0, remaining)
                        remaining -= work
                        current_step_started = True
                    else:
                        state = "Standby"
                        c_cost = params.standby_cost_eur_h
                else:
                    if ok:
                        state = "Lavoro"
                        c_cost = op_cost_for_hour(ts, params)
                        work = min(1.0, remaining)
                        remaining -= work
                    else:
                        state = "Standby"
                        c_cost = params.standby_cost_eur_h

                crane_cost += c_cost

                if remaining <= 1e-9:
                    step_i += 1
                    if step_i < len(steps):
                        remaining = float(steps[step_i].duration_h)
                        current_step_started = False

            logs.append(
                {
                    "timestamp": ts,
                    "state": state,
                    "crane_cost_eur": c_cost,
                    "prod_loss_eur": loss_eur,
                    "step_name": step_name_for_log,
                    "member": m,
                }
            )
            idx += 1

        completion_ts = last_ts

        total = params.mob_demob_eur + crane_cost + prod_loss
        member_results.append(
            {
                "member": m,
                "total_cost_eur": total if not incomplete else np.inf,
                "crane_cost_eur": params.mob_demob_eur + crane_cost if not incomplete else np.inf,
                "prod_loss_eur": prod_loss if not incomplete else np.inf,
                "completion_ts": completion_ts,
                "incomplete": incomplete,
            }
        )
        member_logs.append(pd.DataFrame(logs))

    return {
        "status": "ok",
        "start_day": start_ts,
        "member_results": pd.DataFrame(member_results),
        "member_logs": member_logs,
    }


def compute_daily_summary(all_sims: Dict[pd.Timestamp, Dict]) -> pd.DataFrame:
    rows = []
    for d0, sim in all_sims.items():
        if sim.get("status") != "ok":
            continue
        mr = sim["member_results"]
        costs = mr["total_cost_eur"].to_numpy()

        p10 = safe_percentile(costs, 10)
        p90 = safe_percentile(costs, 90)
        mean = float(np.nanmean(costs[np.isfinite(costs)])) if np.any(np.isfinite(costs)) else np.nan
        spread = p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan

        rows.append(
            {
                "Giorno Inizio (D0)": pd.Timestamp(d0).date(),
                "Costo Min (P10) €": p10,
                "Costo Medio Atteso €": mean,
                "Costo Max (P90) €": p90,
                "Spread (P90-P10) €": spread,
                "Scenari incompleti": int(np.sum(~np.isfinite(costs))),
            }
        )
    out = pd.DataFrame(rows).sort_values("Giorno Inizio (D0)")
    return out.reset_index(drop=True)


def add_confidence(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary.copy()
    spread = s["Spread (P90-P10) €"].to_numpy(dtype=float)
    finite = np.isfinite(spread)
    if finite.sum() >= 2:
        mn = np.nanmin(spread[finite])
        mx = np.nanmax(spread[finite])
        conf = 1.0 - (spread - mn) / (mx - mn + 1e-9)
        conf = np.clip(conf, 0, 1)
    elif finite.sum() == 1:
        conf = np.where(finite, 1.0, np.nan)
    else:
        conf = np.full_like(spread, np.nan)
    s["Livello di Confidenza"] = (conf * 100).round(1)
    return s


def choose_optimal_day(summary_conf: pd.DataFrame, risk_aversion: float = 0.7) -> Tuple[Optional[pd.Timestamp], pd.DataFrame]:
    s = summary_conf.copy()
    mean = s["Costo Medio Atteso €"].to_numpy(dtype=float)
    spread = s["Spread (P90-P10) €"].to_numpy(dtype=float)

    def z(x):
        f = np.isfinite(x)
        if f.sum() < 2:
            out = np.full_like(x, np.nan, dtype=float)
            out[f] = 0.0
            return out
        mu = np.nanmean(x[f])
        sd = np.nanstd(x[f]) + 1e-9
        out = (x - mu) / sd
        out[~f] = np.nan
        return out

    score = z(mean) + float(risk_aversion) * z(spread)
    s["Score (min meglio)"] = score

    if not np.any(np.isfinite(score)):
        return None, s

    best_idx = int(np.nanargmin(score))
    best_day = pd.Timestamp(s.loc[best_idx, "Giorno Inizio (D0)"])
    return best_day, s


# -----------------------------
# PLOTTING (COSTS, GANTT, LOSS)
# -----------------------------
def plot_costs_band(summary: pd.DataFrame) -> go.Figure:
    x = summary["Giorno Inizio (D0)"]
    mean = summary["Costo Medio Atteso €"]
    p10 = summary["Costo Min (P10) €"]
    p90 = summary["Costo Max (P90) €"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.18)",
            name="Incertezza (P10–P90)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines+markers",
            line=dict(color="rgba(99, 102, 241, 1)", width=3),
            marker=dict(size=7),
            name="Costo medio atteso",
        )
    )
    fig.update_layout(
        title="Costo totale vs giorno di inizio (banda P10–P90)",
        xaxis_title="Giorno di inizio D0",
        yaxis_title="€",
        hovermode="x unified",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def aggregate_gantt(member_logs: List[pd.DataFrame]) -> pd.DataFrame:
    if not member_logs:
        return pd.DataFrame()

    all_logs = pd.concat(member_logs, ignore_index=True)
    all_logs = all_logs[["timestamp", "state", "member"]]
    n_members = all_logs["member"].nunique()

    pivot = all_logs.groupby(["timestamp", "state"])["member"].nunique().unstack(fill_value=0).sort_index()
    for col in ["Lavoro", "Standby", "Stop Notte"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Lavoro", "Standby", "Stop Notte"]]
    frac = (pivot / max(1, n_members)).reset_index()
    return frac


def plot_gantt_fraction(frac: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Lavoro"], name="Lavoro", marker_color="rgba(34,197,94,0.85)"))
    fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Standby"], name="Standby", marker_color="rgba(245,158,11,0.85)"))
    fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Stop Notte"], name="Stop Notte", marker_color="rgba(148,163,184,0.85)"))
    fig.update_layout(
        barmode="stack",
        title="Gantt medio (quota scenari per stato, ora per ora)",
        xaxis_title="Ora",
        yaxis_title="Quota scenari",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_daily_prod_loss_band(prod_daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prod_daily["date"], y=prod_daily["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=prod_daily["date"],
            y=prod_daily["p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(239,68,68,0.18)",
            name="Incertezza (P10–P90)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prod_daily["date"],
            y=prod_daily["mean"],
            mode="lines+markers",
            line=dict(color="rgba(239,68,68,1)", width=3),
            marker=dict(size=7),
            name="Perdita media",
        )
    )
    fig.update_layout(
        title="Perdita di fatturato giornaliera (banda P10–P90)",
        xaxis_title="Giorno",
        yaxis_title="€ / giorno",
        hovermode="x unified",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def compute_prod_loss_daily_for_selected(sim: Dict) -> pd.DataFrame:
    member_logs = sim.get("member_logs", [])
    if not member_logs:
        return pd.DataFrame()

    by_member = []
    for mlog in member_logs:
        if mlog.empty:
            continue
        tmp = mlog.copy()
        tmp["date"] = tmp["timestamp"].dt.date
        daily = tmp.groupby("date")["prod_loss_eur"].sum().reset_index()
        daily["member"] = int(tmp["member"].iloc[0]) if "member" in tmp.columns else -1
        by_member.append(daily)

    if not by_member:
        return pd.DataFrame()

    all_daily = pd.concat(by_member, ignore_index=True)
    out = (
        all_daily.groupby("date")["prod_loss_eur"]
        .agg(mean="mean", p10=lambda x: np.percentile(x, 10), p90=lambda x: np.percentile(x, 90))
        .reset_index()
        .sort_values("date")
    )
    return out


# -----------------------------
# UI
# -----------------------------
st.title("WTG Main Component – Decision Making Tool (Open‑Meteo Ensemble)")

with st.sidebar:
    st.header("A) Parametri economici & operativi")
    mob_demob = st.number_input("Mob/Demob (una tantum) [€]", min_value=0.0, value=45000.0, step=1000.0)
    op_std = st.number_input("Costo operativo gru (standard) [€/h]", min_value=0.0, value=1200.0, step=50.0)
    op_fest = st.number_input("Costo operativo gru (festivo/weekend) [€/h]", min_value=0.0, value=1500.0, step=50.0)
    standby = st.number_input("Costo standby (solo in turno) [€/h]", min_value=0.0, value=650.0, step=25.0)
    shift_start = st.text_input("Inizio turno (HH:MM)", value="07:00")
    shift_end = st.text_input("Fine turno (HH:MM)", value="18:00")

    st.divider()
    st.header("B) Open‑Meteo ensemble (live)")
    latitude = st.number_input("Latitudine", value=41.5, format="%.6f")
    longitude = st.number_input("Longitudine", value=15.2, format="%.6f")
    model = st.selectbox(
        "Modello ensemble",
        options=[
            "gfs_seamless",
            "icon_seamless",
            "ecmwf_ifs04",
        ],
        index=0,
        help="Il set di modelli disponibili dipende da Open‑Meteo. Il tool richiede membri ensemble.",
    )
    forecast_days = st.slider("Forecast days (max 16)", min_value=3, max_value=16, value=10, step=1)
    include_gusts = st.toggle("Usa wind_gusts_10m (se disponibili)", value=True)

    st.divider()
    st.header("C) Pianificazione")
    earliest_day = st.date_input("Primo giorno organizzabile (earliest D0)", value=pd.Timestamp.now().date())
    risk_aversion = st.slider("Risk aversion (peso dell'incertezza)", 0.0, 2.0, 0.7, 0.1)

    st.divider()
    st.header("D) Debug")
    use_mock = st.toggle("Usa Mock Data (solo debug)", value=False)


# Validate shift
try:
    _ = to_time(shift_start)
    _ = to_time(shift_end)
except Exception:
    st.error("Formato orari turno non valido. Usa HH:MM (es. 07:00).")
    st.stop()

params = CraneParams(
    mob_demob_eur=float(mob_demob),
    op_cost_std_eur_h=float(op_std),
    op_cost_fest_eur_h=float(op_fest),
    standby_cost_eur_h=float(standby),
    shift_start=shift_start,
    shift_end=shift_end,
)

# Steps table
st.subheader("Attività (step sequenziali)")

default_steps = pd.DataFrame(
    {
        "Step": ["Step 1", "Step 2", "Step 3"],
        "Durata [h]": [8.0, 8.0, 8.0],
        "Wind Threshold [m/s]": [8.0, 8.0, 8.0],
        "Gust Threshold [m/s] (opzionale)": [np.nan, np.nan, np.nan],
        "Finestra minima consecutiva [h]": [3.0, 3.0, 3.0],
    }
)

steps_df = st.data_editor(default_steps, num_rows="dynamic", use_container_width=True, hide_index=True)

steps: List[Step] = []
for _, r in steps_df.iterrows():
    name = str(r["Step"])
    dur = float(r["Durata [h]"])
    wt = float(r["Wind Threshold [m/s]"])
    gt = r["Gust Threshold [m/s] (opzionale)"]
    gt = None if (gt is None or (isinstance(gt, float) and np.isnan(gt))) else float(gt)
    minw = float(r["Finestra minima consecutiva [h]"])
    steps.append(Step(name=name, duration_h=dur, wind_thr=wt, gust_thr=gt, min_seq_h=minw))

total_work_h = float(sum(s.duration_h for s in steps))

# Load meteo
with st.spinner("Caricamento forecast Open‑Meteo..."):
    if use_mock:
        df = generate_mock_open_meteo_ensemble(days=int(forecast_days), n_members=10, include_gusts=include_gusts)
        st.info("Usando Mock Data (debug).")
    else:
        df = fetch_open_meteo_ensemble(
            latitude=float(latitude),
            longitude=float(longitude),
            model=str(model),
            forecast_days=int(forecast_days),
            include_gusts=bool(include_gusts),
            timezone="auto",
        )
        st.success("Forecast caricato da Open‑Meteo Ensemble API. 【1-2f3670】【2-585a0d】")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["date"] = df["timestamp"].dt.date

wind_cols_full, gust_cols_full = detect_members(df)

if not wind_cols_full:
    st.error("Non ho trovato colonne wind ensemble nel dataset.")
    st.stop()

# =============================
# PERFORMANCE SETTINGS
# =============================
col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    use_all_members = st.toggle("Usa tutti gli scenari ensemble", value=False)

with col_perf2:
    n_scenarios = st.number_input(
        "Numero scenari",
        min_value=5,
        max_value=len(wind_cols_full),
        value=10,
        step=1,
        disabled=use_all_members
    )

# selezione effettiva
if use_all_members:
    wind_cols = wind_cols_full
    gust_cols = gust_cols_full
else:
    n = min(len(wind_cols_full), int(n_scenarios))
    wind_cols = wind_cols_full[:n]
    gust_cols = gust_cols_full[:n] if gust_cols_full is not None else None

forecast_start = df["timestamp"].min()
forecast_end = df["timestamp"].max()

# Determine D0 candidates: from earliest_day to last feasible day within forecast
earliest_ts = pd.Timestamp(earliest_day)
earliest_ts = clamp_date_to_forecast(earliest_ts, forecast_start, forecast_end)

# Build candidate days (within forecast dates)
all_days = pd.date_range(start=forecast_start.normalize(), end=forecast_end.normalize(), freq="1D")
all_days = [d for d in all_days if d >= earliest_ts.normalize()]

# Filter: only keep D0 where optimistic completion is <= forecast_end
feasible_days = []
for d0 in all_days:
    comp_opt = optimistic_completion_timestamp(d0, total_work_h, params)
    if comp_opt <= forecast_end:
        feasible_days.append(d0)

# If no feasible days, stop with explanation
if len(feasible_days) == 0:
    st.error(
        "Nessun giorno D0 è analizzabile con l'attuale orizzonte di forecast: "
        "la durata pianificata (anche con meteo perfetto) sfora oltre la fine dei dati disponibili. "
        "Riduci la durata/step, amplia forecast_days (max 16) oppure scegli un D0 più vicino. "
        "【1-2f3670】"
    )
    st.stop()

# Show dataset preview
with st.expander("Anteprima dataset (prime 48 ore)", expanded=False):
    st.dataframe(df.head(48), use_container_width=True)

# Context charts
st.subheader("Contesto dati di input (Power curve, prezzi, produzione prevista)")

# period in exam: from first feasible D0 to min(end of forecast, +3 days)
period_start = feasible_days[0].normalize()
period_end = min(forecast_end, period_start + pd.Timedelta(days=min(7, int(forecast_days))) - pd.Timedelta(seconds=1))
df_view = df[(df["timestamp"] >= period_start) & (df["timestamp"] <= period_end)].copy()

cA, cB = st.columns([1, 1])
with cA:
    st.plotly_chart(plot_power_curve(), use_container_width=True)
with cB:
    st.plotly_chart(plot_prices(df_view), use_container_width=True)

st.plotly_chart(plot_expected_production(df_view, wind_cols), use_container_width=True)

# =============================
# STIMA TEMPO
# =============================
n_days = len(feasible_days)
n_members_used = len(wind_cols)
total_work_h = sum(s.duration_h for s in steps)

# stima empirica
complexity = n_days * n_members_used * total_work_h * 2.5
estimated_seconds = complexity / 60000

col_run1, col_run2 = st.columns([1,2])


with col_run2:
    if estimated_seconds < 1:
        st.info("⏱️ Tempo stimato: <1 secondo")
    else:
        st.info(f"⏱️ Tempo stimato: ~{estimated_seconds:.1f} s")



# Run simulations
st.subheader("Risultati per giorno di inizio (D0)")


run_simulation = st.button(
    "▶️ Esegui simulazione",
    type="primary",
    key="run_simulation_button"
)


sims = {}

if run_simulation:

    with st.spinner("Simulazione stocastica in corso..."):
        for d in feasible_days:
            sims[d] = simulate_single_start_day(
                df=df,
                start_day=d,
                wind_cols=wind_cols,
                gust_cols=gust_cols,
                steps=steps,
                params=params,
                rated_mw=2.0,
            )

    summary = compute_daily_summary(sims)
    summary = add_confidence(summary)
    best_day, scored = choose_optimal_day(summary, risk_aversion=risk_aversion)

else:
    st.info("👉 Premi 'Esegui simulazione' per calcolare i risultati.")
    st.stop()

summary = compute_daily_summary(sims)
summary = add_confidence(summary)
best_day, scored = choose_optimal_day(summary, risk_aversion=risk_aversion)

# KPI
kpi_cols = st.columns([1.2, 1.2, 1.2, 1.4])
if best_day is not None and not scored.empty:
    best_row = scored.loc[scored["Giorno Inizio (D0)"] == best_day.date()].iloc[0]
    kpi_cols[0].metric("Data ottimale (D0)", str(best_day.date()))
    kpi_cols[1].metric("Costo medio atteso", f"{best_row['Costo Medio Atteso €']:,.0f} €")
    kpi_cols[2].metric("P90–P10 (spread)", f"{best_row['Spread (P90-P10) €']:,.0f} €")
    kpi_cols[3].metric("Confidenza", f"{best_row['Livello di Confidenza']:.1f} %")
else:
    st.warning("Non è stato possibile determinare una data ottimale (dati insufficienti).")

display_cols = [
    "Giorno Inizio (D0)",
    "Costo Min (P10) €",
    "Costo Medio Atteso €",
    "Costo Max (P90) €",
    "Livello di Confidenza",
    "Scenari incompleti",
]
st.dataframe(
    summary[display_cols].style.format(
        {
            "Costo Min (P10) €": "{:,.0f}",
            "Costo Medio Atteso €": "{:,.0f}",
            "Costo Max (P90) €": "{:,.0f}",
            "Livello di Confidenza": "{:.1f}",
        }
    ),
    use_container_width=True,
)

st.plotly_chart(plot_costs_band(summary), use_container_width=True)

st.markdown("### Dettaglio per un D0 (Gantt medio + perdite)")
selected = st.selectbox("Seleziona giorno D0", options=[d.date() for d in feasible_days], index=0)
sim_sel = sims.get(pd.Timestamp(selected))
if sim_sel is None or sim_sel.get("status") != "ok":
    st.warning("Simulazione non disponibile per il giorno selezionato.")
else:
    frac = aggregate_gantt(sim_sel["member_logs"])
    if not frac.empty:
        st.plotly_chart(plot_gantt_fraction(frac), use_container_width=True)

    prod_daily = compute_prod_loss_daily_for_selected(sim_sel)
    if not prod_daily.empty:
        st.plotly_chart(plot_daily_prod_loss_band(prod_daily), use_container_width=True)

with st.expander("Perché prima vedevi NaN?", expanded=False):
    st.markdown(
        """
I NaN arrivano quando **tutti** gli scenari per un dato D0 risultano **incompleti** (ossia la simulazione
arriva a fine forecast senza terminare lo step finale).  
Questo succede tipicamente per D0 troppo vicini alla fine del forecast.  

Ora l'app **filtra automaticamente** i D0 e considera solo quelli per cui la durata pianificata
(senza standby, solo vincolo turno/notte) termina **entro** il forecast.
"""
    )

st.caption("✅ Avvio: `streamlit run app.py`")

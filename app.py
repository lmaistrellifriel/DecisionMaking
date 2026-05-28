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
            s = s.replace(",", ".")
            return float(s)
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_bool(v, default=True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not pd.isna(v):
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
    avg_hours_after_d0 = max(1.0, horizon_hours / 2.0)
    ops = n_members * n_days * (avg_hours_after_d0 + 2.0 * total_work_h)
    return ops / 45000.0


def shift_len_hours(shift_start: str, shift_end: str) -> float:
    t0 = pd.to_datetime(shift_start, format="%H:%M")
    t1 = pd.to_datetime(shift_end, format="%H:%M")
    return max(1.0, (t1 - t0).total_seconds() / 3600.0)


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


def plot_power_curve() -> go.Figure:
    wind = np.linspace(0, 30, 300)
    power = np.array([power_curve_mw(w) for w in wind], dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wind, y=power, mode="lines",
        line=dict(color="rgba(34,197,94,1)", width=3),
        name="Power Curve",
        hovertemplate="Vento: %{x:.1f} m/s<br>Potenza: %{y:.2f} MW<extra></extra>"
    ))
    fig.update_layout(
        title="Power Curve (2 MW)",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="Power [MW]",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )
    return fig


# -----------------------------
# PRICES (mock)
# -----------------------------
def generate_price_series(ts: pd.Series, seed: int = 7) -> np.ndarray:
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
# OPEN-METEO: FETCH ENSEMBLE (Ripristinato da allegato)
# -----------------------------
OPEN_METEO_ENSEMBLE_ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _extract_member_cols(hourly: dict, base_name: str) -> List[str]:
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
        raise ValueError("Risposta Open-Meteo non contiene 'hourly.time'.")

    hourly = js["hourly"]
    times = pd.to_datetime(hourly["time"])
    df = pd.DataFrame({"timestamp": times})

    # Ricerca delle colonne dei membri basata sui pattern nativi dell'API Ensemble
    wind_keys = _extract_member_cols(hourly, "wind_speed_80m")
    if not wind_keys:
        candidates = sorted([k for k in hourly.keys() if "wind_speed" in k and "member" in k])
        if candidates:
            root = candidates[0].split("_member")[0]
            wind_keys = _extract_member_cols(hourly, root)

    if not wind_keys:
        raise ValueError(f"Impossibile mappare membri ensemble wind_speed_* per '{model}'.")

    for i, k in enumerate(wind_keys):
        df[f"wind_speed_80m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    gust_keys = _extract_member_cols(hourly, "wind_gusts_10m") if include_gusts else []
    if include_gusts and gust_keys:
        for i, k in enumerate(gust_keys):
            if i < len(wind_keys):
                df[f"wind_gusts_10m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    df["price_eur_mwh"] = generate_price_series(df["timestamp"])
    return df


def generate_mock_open_meteo_ensemble(
    start: str = None,
    days: int = 10,
    n_members: int = 30,
    seed: int = 42,
    include_gusts: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if start is None:
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


def detect_members(df: pd.DataFrame) -> Tuple[List[str], Optional[List[str]]]:
    wind_cols = [c for c in df.columns if c.startswith("wind_speed_80m_member_")]
    wind_cols = sorted(wind_cols, key=lambda c: int(c.split("_")[-1]))
    gust_cols = [c for c in df.columns if c.startswith("wind_gusts_10m_member_")]
    gust_cols = sorted(gust_cols, key=lambda c: int(c.split("_")[-1])) if gust_cols else None
    if gust_cols is not None and len(gust_cols) != len(wind_cols):
        gust_cols = None
    return wind_cols, gust_cols


# -----------------------------
# PLOTS: wind & production
# -----------------------------
def plot_wind_speed_ensemble(df_view: pd.DataFrame, wind_cols: List[str]) -> go.Figure:
    wind_mat = df_view[wind_cols].to_numpy(dtype=float)
    p10 = np.percentile(wind_mat, 10, axis=1)
    p90 = np.percentile(wind_mat, 90, axis=1)
    mean = np.mean(wind_mat, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view["timestamp"], y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df_view["timestamp"], y=p10, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(14,165,233,0.18)", name="P10–P90",
    ))
    fig.add_trace(go.Scatter(
        x=df_view["timestamp"], y=mean, mode="lines",
        line=dict(color="rgba(14,165,233,1)", width=2.5), name="Media",
    ))
    fig.update_layout(
        title="Velocità vento prevista [m/s] (ensemble)",
        xaxis_title="Tempo",
        yaxis_title="m/s",
        template="plotly_white",
        height=320,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_expected_production(df_view: pd.DataFrame, wind_cols: List[str]) -> go.Figure:
    wind_mat = df_view[wind_cols].to_numpy(dtype=float)
    v_power = np.vectorize(power_curve_mw)
    power_mat = v_power(wind_mat)

    p10 = np.percentile(power_mat, 10, axis=1)
    p90 = np.percentile(power_mat, 90, axis=1)
    mean = np.mean(power_mat, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view["timestamp"], y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df_view["timestamp"], y=p10, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(59,130,246,0.18)", name="P10–P90",
    ))
    fig.add_trace(go.Scatter(
        x=df_view["timestamp"], y=mean, mode="lines",
        line=dict(color="rgba(59,130,246,1)", width=2.5), name="Media",
    ))
    fig.update_layout(
        title="Produzione prevista [MW] (ensemble + power curve)",
        xaxis_title="Tempo",
        yaxis_title="MW",
        template="plotly_white",
        height=320,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# -----------------------------
# SIMULATION CORE (Monte Carlo / Cost Only)
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


def check_min_window(df: pd.DataFrame, start_idx: int, wind_col: str, gust_col: Optional[str], step: Step, params: CraneParams) -> bool:
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


def count_remaining_work_hours(df: pd.DataFrame, d0_ts: pd.Timestamp, params: CraneParams) -> int:
    mask = (df["timestamp"] >= d0_ts) & (df["timestamp"] <= df["timestamp"].iloc[-1])
    ts = df.loc[mask, "timestamp"]
    if ts.empty:
        return 0
    return int(np.sum(ts.apply(lambda x: in_work_shift(x, params.shift_start, params.shift_end)).to_numpy(dtype=bool)))


def simulate_single_start_day_cost(
    df: pd.DataFrame,
    start_day: pd.Timestamp,
    wind_cols: List[str],
    gust_cols: Optional[List[str]],
    steps: List[Step],
    params: CraneParams,
    rated_mw: float = 2.0
) -> Dict:
    df = df.sort_values("timestamp").reset_index(drop=True)
    d0_ts = pd.Timestamp(start_day.date())

    d0_idx = int(df["timestamp"].searchsorted(d0_ts))
    if d0_idx >= len(df):
        d0_idx = len(df) - 1
        d0_ts = df["timestamp"].iloc[-1]
    else:
        d0_ts = df.at[d0_idx, "timestamp"]

    mob_demob_apply = params.mob_demob_eur if any(s.requires_crane for s in steps) else 0.0

    member_rows = []
    member_logs = []

    for m, wind_col in enumerate(wind_cols):
        gust_col = gust_cols[m] if gust_cols is not None else None

        step_i = 0
        remaining = float(steps[0].duration_h) if steps else 0.0
        current_step_started = False

        crane_cost = 0.0
        lost_revenue = 0.0

        idx = d0_idx
        last_ts = df["timestamp"].iloc[d0_idx]
        logs = []

        while idx < len(df) and step_i < len(steps):
            ts = df.at[idx, "timestamp"]
            last_ts = ts
            step = steps[step_i]

            w = df.at[idx, wind_col]
            g = df.at[idx, gust_col] if gust_col is not None else np.nan

            p_mw = power_curve_mw(w, rated_mw=rated_mw)
            price = float(df.at[idx, "price_eur_mwh"])
            loss_eur = p_mw * price
            lost_revenue += loss_eur

            crane_present = any(s.requires_crane for s in steps[step_i:])
            c_cost = 0.0

            if not in_work_shift(ts, params.shift_start, params.shift_end):
                state = "Stop Notte"
                c_cost = 0.0
            else:
                ok = float(w) < float(step.wind_thr)
                if gust_col is not None and step.gust_thr is not None:
                    ok = ok and (float(g) < float(step.gust_thr))

                if not current_step_started:
                    window_ok = check_min_window(df, idx, wind_col, gust_col, step, params)
                else:
                    window_ok = True

                if ok and window_ok:
                    state = "Lavoro"
                    work = min(1.0, remaining)
                    remaining -= work
                    current_step_started = True
                    if crane_present:
                        c_cost = op_cost_for_hour(ts, params)
                else:
                    state = "Standby" if crane_present else "Attesa (no gru)"
                    if crane_present:
                        c_cost = params.standby_cost_eur_h

                crane_cost += c_cost

                if remaining <= 1e-9:
                    step_i += 1
                    if step_i < len(steps):
                        remaining = float(steps[step_i].duration_h)
                        current_step_started = False

            logs.append({
                "timestamp": ts,
                "state": state,
                "crane_cost_eur": c_cost,
                "prod_loss_eur": loss_eur,
                "step_name": step.name,
                "member": m,
                "crane_present": crane_present,
            })
            idx += 1

        partial = (step_i < len(steps))
        total_cost = mob_demob_apply + crane_cost + lost_revenue

        member_rows.append({
            "member": m,
            "total_cost_eur": total_cost,
            "mob_demob_eur": mob_demob_apply,
            "crane_cost_eur": crane_cost,
            "lost_revenue_eur": lost_revenue,
            "partial": partial,
            "completion_ts": last_ts,
        })
        member_logs.append(pd.DataFrame(logs))

    return {
        "status": "ok",
        "start_day": d0_ts,
        "member_results": pd.DataFrame(member_rows),
        "member_logs": member_logs
    }


def compute_daily_summary_cost(all_sims: Dict[pd.Timestamp, Dict], structural_infeasible: Dict[pd.Timestamp, bool]) -> pd.DataFrame:
    rows = []
    for d0, sim in all_sims.items():
        if sim.get("status") != "ok":
            continue
        mr = sim["member_results"]
        costs = mr["total_cost_eur"].to_numpy(dtype=float)

        total_scenarios = len(mr)
        success = int(np.sum(~mr["partial"].to_numpy(dtype=bool))) if total_scenarios else 0
        prob_success = (success / total_scenarios) * 100.0 if total_scenarios else 0.0

        p10 = safe_percentile(costs, 10)
        p90 = safe_percentile(costs, 90)
        mean = float(np.nanmean(costs[np.isfinite(costs)])) if np.any(np.isfinite(costs)) else np.nan
        spread = p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan

        rows.append({
            "Giorno Inizio (D0)": pd.Timestamp(d0).date(),
            "Probabilità Successo (%)": prob_success,
            "Costo P10 €": p10,
            "Costo Medio €": mean,
            "Costo P90 €": p90,
            "Spread (P90-P10) €": spread,
            "Strutturalmente impossibile": bool(structural_infeasible.get(pd.Timestamp(d0), False)),
        })

    return pd.DataFrame(rows).sort_values("Giorno Inizio (D0)").reset_index(drop=True)


def choose_optimal_day_cost(summary: pd.DataFrame, last_possible_start: Optional[pd.Timestamp], risk_aversion: float = 0.7) -> Tuple[Optional[pd.Timestamp], pd.DataFrame, str]:
    s = summary.copy()
    if s.empty:
        return None, s, "Nessun dato"

    mean = s["Costo Medio €"].to_numpy(dtype=float)
    spread = np.nan_to_num(s["Spread (P90-P10) €"].to_numpy(dtype=float), nan=0.0)
    score = mean + float(risk_aversion) * spread
    s["Score (min meglio)"] = score

    if last_possible_start is None:
        return None, s, "Finestra temporale insufficiente (nessun D0 completabile per ore turno disponibili)"

    candidates = s[pd.to_datetime(s["Giorno Inizio (D0)"]) <= pd.Timestamp(last_possible_start.date())].copy()
    if candidates.empty:
        return None, s, "La data minima D0 è oltre l’ultimo giorno utile per iniziare"

    if np.nanmax(candidates["Probabilità Successo (%)"].to_numpy(dtype=float)) <= 0.0:
        return None, s, "Probabilità di completamento = 0% per tutti i D0 candidabili"

    best_idx = int(np.nanargmin(candidates["Score (min meglio)"].to_numpy(dtype=float)))
    best_day = pd.Timestamp(candidates.iloc[best_idx]["Giorno Inizio (D0)"])
    return best_day, s, ""


def plot_cost_candles(summary_for_plot: pd.DataFrame) -> go.Figure:
    if summary_for_plot.empty:
        return go.Figure()

    dfp = summary_for_plot.copy().sort_values("Giorno Inizio (D0)")
    x = dfp["Giorno Inizio (D0)"].astype(str)
    p10 = dfp["Costo P10 €"].to_numpy(dtype=float)
    p90 = dfp["Costo P90 €"].to_numpy(dtype=float)
    mean = dfp["Costo Medio €"].to_numpy(dtype=float)

    body = p90 - p10

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=body, base=p10,
        marker=dict(color="rgba(99,102,241,0.35)"),
        name="Intervallo P10–P90",
        hovertemplate="D0: %{x}<br>P10: %{base:,.0f} €<br>P90: %{y+base:,.0f} €<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines+markers",
        line=dict(color="rgba(99,102,241,1)", width=2),
        marker=dict(size=6, color="rgba(99,102,241,1)"),
        name="Costo medio",
        hovertemplate="D0: %{x}<br>Costo medio: %{y:,.0f} €<extra></extra>"
    ))
    fig.update_layout(
        title="Costo totale vs D0 (candela P10–P90) — esclusi solo D0 strutturalmente impossibili",
        xaxis_title="Giorno di inizio D0",
        yaxis_title="Costo Totale (€)",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        barmode="overlay",
    )
    return fig


def aggregate_gantt(member_logs: List[pd.DataFrame]) -> pd.DataFrame:
    if not member_logs:
        return pd.DataFrame()
    all_logs = pd.concat(member_logs, ignore_index=True)
    n_members = all_logs["member"].nunique()
    pivot = all_logs.groupby(["timestamp", "state"])["member"].nunique().unstack(fill_value=0).sort_index()
    for col in ["Lavoro", "Standby", "Stop Notte", "Attesa (no gru)"]:
        if col not in pivot.columns:
            pivot[col] = 0
    frac = (pivot / max(1, n_members)).reset_index()
    return frac


def plot_gantt_fraction(frac: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, color in [
        ("Lavoro", "rgba(34,197,94,0.85)"),
        ("Standby", "rgba(245,158,11,0.85)"),
        ("Attesa (no gru)", "rgba(59,130,246,0.30)"),
        ("Stop Notte", "rgba(148,163,184,0.85)")
    ]:
        if name in frac.columns:
            fig.add_trace(go.Bar(x=frac["timestamp"], y=frac[name], name=name, marker_color=color))
    fig.update_layout(
        barmode="stack",
        title="Gantt medio (quota scenari per stato orario)",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("WTG Main Component – Cost Minimizer (Stochastic)")

with st.sidebar:
    st.header("A) Parametri economici & operativi")
    mob_demob = st.number_input("Mob/Demob [€]", value=45000.0)
    op_std = st.number_input("Costo operativo standard [€/h]", value=1200.0)
    op_fest = st.number_input("Costo operativo weekend [€/h]", value=1500.0)
    standby = st.number_input("Costo standby [€/h]", value=650.0)
    shift_start = st.text_input("Inizio turno", value="07:00")
    shift_end = st.text_input("Fine turno", value="18:00")

    st.divider()
    st.header("B) Open‑Meteo ensemble")
    latitude = st.number_input("Latitudine", value=41.5)
    longitude = st.number_input("Longitudine", value=15.2)
    model = st.selectbox("Modello", options=["gfs_seamless", "icon_seamless", "ecmwf_ifs04"])
    forecast_days = st.slider("Giorni forecast (richiesti)", 3, 16, 10)
    include_gusts = st.toggle("Usa raffiche", value=True)

    st.divider()
    st.header("C) Campionamento")
    use_all_members = st.toggle("Usa tutti i membri", value=False)
    n_members_input = st.number_input("Membri da usare", min_value=1, value=10, disabled=use_all_members)

    st.divider()
    st.header("D) Ottimizzazione")
    earliest_day = st.date_input("Data minima D0", value=pd.Timestamp.now().date())
    risk_aversion = st.slider("Risk aversion (penalità spread)", 0.0, 3.0, 0.7)

    st.divider()
    st.header("E) Debug")
    use_mock = st.toggle("Mock Data (debug)", value=False)

try:
    t0 = to_time(shift_start)
    t1 = to_time(shift_end)
    if t0 >= t1:
        st.error("Orari turno non validi: l'inizio deve essere precedente alla fine.")
        st.stop()
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
    dur = safe_float(r.get("Durata [h]"))
    wt = safe_float(r.get("Wind Threshold [m/s]"))
    minw = safe_float(r.get("Finestra minima consecutiva [h]"), default=0.0)
    gt_raw = r.get("Gust Threshold [m/s] (opzionale)")
    gt = None if (gt_raw is None or pd.isna(gt_raw)) else safe_float(gt_raw)
    req = safe_bool(r.get("Richiede Gru"), default=True)

    if name == "" and dur is None and wt is None:
        continue
    if dur is None or wt is None or dur <= 0 or wt <= 0:
        continue

    steps.append(Step(
        name=name if name else "Step",
        duration_h=float(dur),
        wind_thr=float(wt),
        gust_thr=gt,
        min_seq_h=float(minw or 0.0),
        requires_crane=req
    ))

if not steps:
    st.error("Inserisci almeno uno step valido (Durata > 0 e Wind Threshold > 0).")
    st.stop()

required_work_h = float(sum(s.duration_h for s in steps))

# Caricamento dati meteo
with st.spinner("Caricamento dati meteo..."):
    if use_mock:
        df = generate_mock_open_meteo_ensemble(days=int(forecast_days), n_members=30, include_gusts=include_gusts)
        st.info("Usando Mock Data (debug).")
    else:
        df = fetch_open_meteo_ensemble(latitude, longitude, model, forecast_days, include_gusts)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

forecast_start = df["timestamp"].min()
forecast_end = df["timestamp"].max()
actual_forecast_days = int((forecast_end.normalize() - forecast_start.normalize()).days) + 1

if actual_forecast_days < int(forecast_days) and not use_mock:
    st.warning(
        f"⚠️ Il modello selezionato ha restituito {actual_forecast_days} giorni di forecast "
        f"(richiesti: {int(forecast_days)}). Verrà usato SOLO l'orizzonte reale disponibile."
    )
st.caption(
    f"Orizzonte reale disponibile: {forecast_start.strftime('%d/%m/%Y %H:%M')} → "
    f"{forecast_end.strftime('%d/%m/%Y %H:%M')} ({actual_forecast_days} giorni)"
)

wind_cols_all, gust_cols_all = detect_members(df)
if not wind_cols_all:
    st.error("Nessun membro ensemble trovato nel dataset.")
    st.stop()

wind_cols_use = wind_cols_all if use_all_members else wind_cols_all[:min(len(wind_cols_all), int(n_members_input))]
gust_cols_use = gust_cols_all[:len(wind_cols_use)] if gust_cols_all else None

horizon_start = max(forecast_start.normalize(), pd.Timestamp(earliest_day))

all_days = pd.date_range(start=horizon_start.normalize(), end=forecast_end.normalize(), freq="1D").to_list()
if not all_days:
    st.error("Nessun giorno D0 disponibile nell'orizzonte reale.")
    st.stop()

# Layout grafici (sempre visibili)
st.subheader("Contesto meteo & produzione (sempre visibile)")
preview_end = min(forecast_end, horizon_start + pd.Timedelta(days=5) - pd.Timedelta(seconds=1))
df_view = df[(df["timestamp"] >= horizon_start) & (df["timestamp"] <= preview_end)].copy()

c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(plot_power_curve(), use_container_width=True)
with c2:
    st.plotly_chart(plot_wind_speed_ensemble(df_view, wind_cols_use), use_container_width=True)
st.plotly_chart(plot_expected_production(df_view, wind_cols_use), use_container_width=True)

structural_infeasible = {}
available_work_h_map = {}

for d0 in all_days:
    d0_ts = pd.Timestamp(d0.date())
    idx0 = int(df["timestamp"].searchsorted(d0_ts))
    if idx0 >= len(df):
        structural_infeasible[pd.Timestamp(d0)] = True
        available_work_h_map[pd.Timestamp(d0)] = 0
        continue
    d0_aligned = df.at[idx0, "timestamp"]
    avail = count_remaining_work_hours(df, d0_aligned, params)
    available_work_h_map[pd.Timestamp(d0)] = avail
    structural_infeasible[pd.Timestamp(d0)] = (avail < required_work_h)

feasible_days_only = [pd.Timestamp(d) for d in all_days if not structural_infeasible.get(pd.Timestamp(d), False)]
last_possible_start = pd.Timestamp(feasible_days_only[-1]) if feasible_days_only else None

if last_possible_start is not None:
    st.caption(f"Ultimo D0 possibile (ore turno sufficienti fino a fine forecast reale): {last_possible_start.date()}")
else:
    st.caption("Ultimo D0 possibile: nessuno (ore turno insufficienti nell’orizzonte reale)")

st.subheader("Simulazione stocastica (Costi totali)")

horizon_hours = int((forecast_end - horizon_start).total_seconds() / 3600) + 1
est_sec = heuristic_estimate_seconds(len(wind_cols_use), len(all_days), horizon_hours, required_work_h)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Giorni D0 simulati", len(all_days))
m2.metric("Membri ensemble usati", len(wind_cols_use))
m3.metric("Orizzonte (ore)", horizon_hours)
m4.metric("Simulazioni totali", len(all_days) * len(wind_cols_use))
st.info(f"⏱️ Tempo stimato: {format_time_estimate(est_sec)} ({len(wind_cols_use)} membri × {len(all_days)} giorni)")

cur_hash = hashlib.md5(json.dumps({
    "m": len(wind_cols_use),
    "d": [str(x.date()) for x in all_days],
    "steps": [(s.name, s.duration_h, s.wind_thr, s.gust_thr, s.min_seq_h, s.requires_crane) for s in steps],
    "p": [mob_demob, op_std, op_fest, standby, shift_start, shift_end],
    "ra": risk_aversion,
    "hs": str(horizon_start),
    "fe": str(forecast_end),
}, sort_keys=True).encode()).hexdigest()

if "sims_cost" not in st.session_state:
    st.session_state["sims_cost"] = None
if "sims_hash_cost" not in st.session_state:
    st.session_state["sims_hash_cost"] = None

if st.session_state["sims_cost"] is not None and st.session_state["sims_hash_cost"] != cur_hash:
    st.warning("⚠️ Parametri cambiati dall'ultima esecuzione. Premi il pulsante per aggiornare.")

run_clicked = st.button("▶ Esegui simulazione", type="primary", use_container_width=True, key="btn_run_cost_main")

if run_clicked:
    sims = {}
    prog = st.progress(0.0, text="Avvio...")
    t_start = time.perf_counter()
    for i, d0 in enumerate(all_days):
        sims[pd.Timestamp(d0)] = simulate_single_start_day_cost(
            df=df,
            start_day=pd.Timestamp(d0),
            wind_cols=wind_cols_use,
            gust_cols=gust_cols_use,
            steps=steps,
            params=params
        )
        prog.progress((i + 1) / len(all_days), text=f"D0 {i+1}/{len(all_days)}")
    prog.empty()
    st.session_state["sims_cost"] = sims
    st.session_state["sims_hash_cost"] = cur_hash
    st.success(f"✅ Simulazione completata in {format_time_estimate(time.perf_counter() - t_start)}.")

if st.session_state["sims_cost"] is None:
    st.info("👆 Premi **Esegui simulazione** per ottenere i risultati.")
    st.stop()

sims = st.session_state["sims_cost"]
summary = compute_daily_summary_cost(sims, structural_infeasible)
best_d0, scored, reason_none = choose_optimal_day_cost(summary, last_possible_start, risk_aversion=risk_aversion)

st.header("Risultati (Costi totali + Probabilità di completamento)")

c1, c2, c3, c4 = st.columns(4)
if best_d0 is None:
    c1.metric("Miglior D0", "Nessun D0 ottimo")
    c2.metric("Motivo", reason_none if reason_none else "n.d.")
    c3.metric("Ore richieste", f"{required_work_h:.0f} h turno")
    c4.metric("Ultimo D0 possibile", str(last_possible_start.date()) if last_possible_start is not None else "Nessuno")
else:
    rb = scored[scored["Giorno Inizio (D0)"] == best_d0.date()].iloc[0]
    c1.metric("Miglior D0 (min costo medio)", best_d0.strftime("%d/%m/%Y"))
    c2.metric("Costo medio atteso", f"{rb['Costo Medio €']:,.0f} €")
    c3.metric("Probabilità successo", f"{rb['Probabilità Successo (%)']:.1f} %")
    c4.metric("Ultimo D0 possibile", str(last_possible_start.date()) if last_possible_start is not None else "Nessuno")

st.subheader("Tabella completa (tutti i D0 simulati, formattazione uniforme)")
st.dataframe(
    scored[[
        "Giorno Inizio (D0)",
        "Probabilità Successo (%)",
        "Costo P10 €",
        "Costo Medio €",
        "Costo P90 €",
        "Spread (P90-P10) €",
        "Strutturalmente impossibile",
    ]].style.format({
        "Probabilità Successo (%)": "{:.1f}",
        "Costo P10 €": "{:,.0f}",
        "Costo Medio €": "{:,.0f}",
        "Costo P90 €": "{:,.0f}",
        "Spread (P90-P10) €": "{:,.0f}",
    }),
    use_container_width=True,
    hide_index=True
)

st.subheader("Costo totale vs D0 (grafico a candele: esclusi solo i D0 strutturalmente impossibili)")
plot_df = scored[~scored["Strutturalmente impossibile"]].copy()
if plot_df.empty:
    st.warning("Nessun D0 è plottabile: tutti i D0 risultano strutturalmente impossibili (ore turno insufficienti).")
else:
    st.plotly_chart(plot_cost_candles(plot_df), use_container_width=True)

st.subheader("Dettaglio D0 selezionato (anche se non plottabile nel grafico a candele)")
selected = st.selectbox("Seleziona D0", [d.date() for d in all_days], key="sel_d0_detail")
sim_sel = sims.get(pd.Timestamp(selected))

sel_ts = pd.Timestamp(selected)
sel_infeasible = bool(structural_infeasible.get(sel_ts, False))
sel_avail = available_work_h_map.get(sel_ts, 0)

st.caption(
    f"D0 selezionato: {'STRUTTURALMENTE IMPOSSIBILE' if sel_infeasible else 'strutturalmente completabile'} "
    f"(ore turno disponibili: {sel_avail} vs richieste: {required_work_h:.0f})"
)

if sim_sel and sim_sel.get("status") == "ok":
    mr = sim_sel["member_results"]
    cost_arr = mr["total_cost_eur"].to_numpy(dtype=float)
    success_prob = 100.0 * float(np.mean(~mr["partial"].to_numpy(dtype=bool))) if len(mr) else 0.0

    dcols = st.columns(4)
    dcols[0].metric("Probabilità successo", f"{success_prob:.1f} %")
    dcols[1].metric("Costo medio", f"{np.nanmean(cost_arr):,.0f} €")
    dcols[2].metric("Costo P10", f"{np.percentile(cost_arr, 10):,.0f} €")
    dcols[3].metric("Costo P90", f"{np.percentile(cost_arr, 90):,.0f} €")

    frac = aggregate_gantt(sim_sel["member_logs"])
    if not frac.empty:
        st.plotly_chart(plot_gantt_fraction(frac), use_container_width=True)
else:
    st.info("Nessun dettaglio disponibile per il D0 selezionato.")

st.caption("✅ Avvio: `streamlit run app.py`")

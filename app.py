
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
def to_time(hhmm: str) -> pd.Timestamp:
    # returns a dummy timestamp with given time (date irrelevant)
    return pd.to_datetime(hhmm).time()

def is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5  # 5=Sat,6=Sun

def in_work_shift(ts: pd.Timestamp, shift_start: str, shift_end: str) -> bool:
    """
    Work shift defined within the same calendar day.
    Assumes shift_start < shift_end (e.g. 07:00 to 18:00).
    """
    t = ts.time()
    return (t >= to_time(shift_start)) and (t < to_time(shift_end))

def shift_end_timestamp(ts: pd.Timestamp, shift_end: str) -> pd.Timestamp:
    return pd.Timestamp(ts.date()) + pd.to_timedelta(f"{shift_end}:00")

def safe_percentile(a: np.ndarray, q: float) -> float:
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.percentile(a, q))

# -----------------------------
# POWER CURVE (2 MW reference)
# -----------------------------
def power_curve_mw(wind_ms: float,
                   rated_mw: float = 2.0,
                   cut_in: float = 3.0,
                   rated: float = 12.0,
                   cut_out: float = 25.0) -> float:
    """
    Simple generic curve:
    - 0 below cut-in
    - cubic ramp to rated
    - constant rated to cut-out
    - 0 above cut-out
    """
    if wind_ms is None or np.isnan(wind_ms):
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

# -----------------------------
# MOCK DATA GENERATION
# -----------------------------
def generate_mock_open_meteo_ensemble(
    start: str = "2026-03-09 00:00",
    days: int = 10,
    n_members: int = 10,
    seed: int = 42,
    include_gusts: bool = True,
    latitude: float = 41.5,
    longitude: float = 15.2,
) -> pd.DataFrame:
    """
    Produce an hourly dataset shaped like an Open-Meteo ensemble export:
    - timestamp
    - price_eur_mwh (mock GME-like intraday shape)
    - wind_speed_80m_member_{i}
    - optional wind_gusts_10m_member_{i}
    Uncertainty increases after day 3.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=days * 24, freq="1h")
    df = pd.DataFrame({"timestamp": idx})
    df["latitude"] = latitude
    df["longitude"] = longitude

    # Price profile: realistic intraday shape 60-120 €/MWh (demo only)
    hour = df["timestamp"].dt.hour.values
    dow = df["timestamp"].dt.dayofweek.values

    # Base daily curve: night low, morning ramp, evening peak
    base = (
        70
        + 10 * np.sin((hour - 6) / 24 * 2 * np.pi)     # smooth day shape
        + 25 * np.exp(-((hour - 20) ** 2) / (2 * 2.8 ** 2))  # evening peak
        + 10 * np.exp(-((hour - 9) ** 2) / (2 * 2.8 ** 2))   # morning bump
    )

    # Weekend slightly lower in this mock
    weekend_factor = np.where(dow >= 5, -7, 0)
    noise = rng.normal(0, 3.0, size=len(df))
    price = np.clip(base + weekend_factor + noise, 60, 120)
    df["price_eur_mwh"] = np.round(price, 2)

    # Wind "truth" baseline pattern (hourly + multi-day modulation)
    day_index = ((df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 86400).values
    di = np.floor(day_index).astype(int)
    w_base = (
        7.5
        + 1.8 * np.sin((hour) / 24 * 2 * np.pi)          # diurnal
        + 1.2 * np.sin((day_index) / 7 * 2 * np.pi)      # weekly-ish
        + 0.6 * np.sin((hour - 14) / 24 * 4 * np.pi)     # more variability
    )

    # Uncertainty: low first 3 days, then grows
    sigma = np.where(di < 3, 0.9, 0.9 + (di - 2) * (3.2 / max(1, (days - 3))))
    sigma = np.clip(sigma, 0.9, 4.0)

    for m in range(n_members):
        member_bias = rng.normal(0, 0.4)  # each member has small bias
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
    """
    Detect ensemble member columns.
    Returns:
      wind_cols: list ordered by member index
      gust_cols: list ordered by member index (or None if absent)
    """
    wind_cols = [c for c in df.columns if c.startswith("wind_speed_80m_member_")]
    if not wind_cols:
        # fallback: allow wind_80m_member_ or wind_speed_member_
        wind_cols = [c for c in df.columns if "wind" in c.lower() and "member" in c.lower() and "speed" in c.lower()]

    # sort by trailing integer if possible
    def key_member(col: str):
        try:
            return int(col.split("_")[-1])
        except Exception:
            return 999999

    wind_cols = sorted(wind_cols, key=key_member)

    gust_cols = [c for c in df.columns if c.startswith("wind_gusts_10m_member_")]
    gust_cols = sorted(gust_cols, key=key_member) if gust_cols else None

    # Align gust members count if partial
    if gust_cols is not None and len(gust_cols) != len(wind_cols):
        # if mismatch, ignore gusts to avoid wrong mapping
        gust_cols = None

    return wind_cols, gust_cols

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
    """
    Check if there are at least min_seq_h consecutive "good" hours
    starting from current hour within the SAME shift window.
    Conservative: uses ceil(min_seq_h).
    """
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

def simulate_single_start_day(
    df: pd.DataFrame,
    start_day: pd.Timestamp,
    wind_cols: List[str],
    gust_cols: Optional[List[str]],
    steps: List[Step],
    params: CraneParams,
    rated_mw: float = 2.0,
) -> Dict:
    """
    Simulate for a given D0 (start_day date at 00:00) across all ensemble members.
    Returns per-member totals + per-member hourly logs (for gantt/production aggregation).
    """
    start_ts = pd.Timestamp(start_day.date())  # 00:00
    df = df.sort_values("timestamp").reset_index(drop=True)

    # locate first index >= start_ts
    start_idx = int(df["timestamp"].searchsorted(start_ts))
    if start_idx >= len(df):
        return {"status": "out_of_range"}

    member_results = []
    member_logs = []  # list of DataFrames per member: timestamp, state, crane_cost, prod_loss, step_name

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

        # simulate until all steps complete or data ends
        while step_i < len(steps):
            if idx >= len(df):
                incomplete = True
                break

            ts = df.at[idx, "timestamp"]
            step = steps[step_i]

            # meteo values
            w = df.at[idx, wind_col]
            g = df.at[idx, gust_col] if gust_col is not None else np.nan

            # always compute production loss H24
            p_mw = power_curve_mw(w, rated_mw=rated_mw)
            price = float(df.at[idx, "price_eur_mwh"]) if "price_eur_mwh" in df.columns else 0.0
            loss_eur = p_mw * price  # MWh in one hour * €/MWh
            prod_loss += loss_eur

            if not in_work_shift(ts, params.shift_start, params.shift_end):
                state = "Stop Notte"
                c_cost = 0.0
                # no progress
            else:
                # check if meteo is good this hour
                ok = float(w) < float(step.wind_thr)
                if gust_col is not None and step.gust_thr is not None:
                    ok = ok and (float(g) < float(step.gust_thr))

                if not current_step_started:
                    # need minimum window from this hour
                    window_ok = check_min_window(df, idx, wind_col, gust_col, step, params)
                    if window_ok and ok:
                        state = "Lavoro"
                        c_cost = op_cost_for_hour(ts, params)
                        # allow fractional progress
                        work = min(1.0, remaining)
                        remaining -= work
                        current_step_started = True
                    else:
                        state = "Standby"
                        c_cost = params.standby_cost_eur_h
                else:
                    # step already started
                    if ok:
                        state = "Lavoro"
                        c_cost = op_cost_for_hour(ts, params)
                        work = min(1.0, remaining)
                        remaining -= work
                    else:
                        state = "Standby"
                        c_cost = params.standby_cost_eur_h

                crane_cost += c_cost

                # step complete?
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
                    "step_name": steps[step_i].name if step_i < len(steps) else "Completed",
                    "member": m,
                }
            )
            idx += 1

        completion_ts = df.at[min(idx, len(df) - 1), "timestamp"] if len(df) > 0 else start_ts

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

def choose_optimal_day(summary_conf: pd.DataFrame, risk_aversion: float = 0.5) -> Tuple[Optional[pd.Timestamp], pd.DataFrame]:
    """
    Optimize a score combining expected cost and uncertainty spread:
    score = z(mean_cost) + risk_aversion * z(spread)
    Lower score is better.
    """
    s = summary_conf.copy()
    mean = s["Costo Medio Atteso €"].to_numpy(dtype=float)
    spread = s["Spread (P90-P10) €"].to_numpy(dtype=float)

    def z(x):
        f = np.isfinite(x)
        if f.sum() < 2:
            return np.zeros_like(x)
        mu = np.nanmean(x[f])
        sd = np.nanstd(x[f]) + 1e-9
        out = (x - mu) / sd
        out[~f] = np.nan
        return out

    score = z(mean) + float(risk_aversion) * z(spread)
    s["Score (min meglio)"] = score
    best_idx = int(np.nanargmin(score)) if np.any(np.isfinite(score)) else None

    if best_idx is None:
        return None, s

    best_day = pd.Timestamp(s.loc[best_idx, "Giorno Inizio (D0)"])
    return best_day, s

# -----------------------------
# PLOTTING
# -----------------------------
def plot_costs_band(summary: pd.DataFrame) -> go.Figure:
    x = summary["Giorno Inizio (D0)"]
    mean = summary["Costo Medio Atteso €"]
    p10 = summary["Costo Min (P10) €"]
    p90 = summary["Costo Max (P90) €"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p90, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
        name="P90"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p10, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(99, 102, 241, 0.18)",
        showlegend=True, name="Incertezza (P10–P90)"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines+markers",
        line=dict(color="rgba(99, 102, 241, 1)", width=3),
        marker=dict(size=7),
        name="Costo medio atteso"
    ))
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
    """
    From per-member hourly logs produce fractions per state by timestamp.
    """
    if not member_logs:
        return pd.DataFrame()

    all_logs = pd.concat(member_logs, ignore_index=True)
    # keep only relevant columns
    all_logs = all_logs[["timestamp", "state", "member"]]
    n_members = all_logs["member"].nunique()

    pivot = (
        all_logs
        .groupby(["timestamp", "state"])["member"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    for col in ["Lavoro", "Standby", "Stop Notte"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Lavoro", "Standby", "Stop Notte"]]
    frac = pivot / max(1, n_members)
    frac = frac.reset_index()
    return frac

def plot_gantt_fraction(frac: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=frac["timestamp"], y=frac["Lavoro"],
        name="Lavoro", marker_color="rgba(34,197,94,0.85)"
    ))
    fig.add_trace(go.Bar(
        x=frac["timestamp"], y=frac["Standby"],
        name="Standby", marker_color="rgba(245,158,11,0.85)"
    ))
    fig.add_trace(go.Bar(
        x=frac["timestamp"], y=frac["Stop Notte"],
        name="Stop Notte", marker_color="rgba(148,163,184,0.85)"
    ))
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
    """
    prod_daily columns: date, mean, p10, p90
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prod_daily["date"], y=prod_daily["p90"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=prod_daily["date"], y=prod_daily["p10"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(239,68,68,0.18)", name="Incertezza (P10–P90)"
    ))
    fig.add_trace(go.Scatter(
        x=prod_daily["date"], y=prod_daily["mean"], mode="lines+markers",
        line=dict(color="rgba(239,68,68,1)", width=3),
        marker=dict(size=7),
        name="Perdita media"
    ))
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
    """
    For a single D0 simulation: build distribution of daily revenue loss across members.
    """
    member_logs = sim.get("member_logs", [])
    if not member_logs:
        return pd.DataFrame()

    by_member = []
    for mlog in member_logs:
        if mlog.empty:
            continue
        tmp = mlog.copy()
        tmp["date"] = tmp["timestamp"].dt.date
        # prod_loss_eur already hourly
        daily = tmp.groupby("date")["prod_loss_eur"].sum().reset_index()
        daily["member"] = int(tmp["member"].iloc[0]) if "member" in tmp.columns else -1
        by_member.append(daily)

    if not by_member:
        return pd.DataFrame()

    all_daily = pd.concat(by_member, ignore_index=True)
    # compute per-day percentiles across members
    out = (
        all_daily
        .groupby("date")["prod_loss_eur"]
        .agg(
            mean="mean",
            p10=lambda x: np.percentile(x, 10),
            p90=lambda x: np.percentile(x, 90),
        )
        .reset_index()
        .sort_values("date")
    )
    return out

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.title("WTG Main Component – Decision Making Tool (stocastico su Ensemble)")

with st.sidebar:
    st.header("1) Parametri economici & operativi")

    mob_demob = st.number_input("Mob/Demob (una tantum) [€]", min_value=0.0, value=45000.0, step=1000.0)
    op_std = st.number_input("Costo operativo gru (standard) [€/h]", min_value=0.0, value=1200.0, step=50.0)
    op_fest = st.number_input("Costo operativo gru (festivo/weekend) [€/h]", min_value=0.0, value=1500.0, step=50.0)
    standby = st.number_input("Costo standby (solo in turno) [€/h]", min_value=0.0, value=650.0, step=25.0)

    st.divider()
    shift_start = st.text_input("Inizio turno (HH:MM)", value="07:00")
    shift_end = st.text_input("Fine turno (HH:MM)", value="18:00")

    st.divider()
    st.header("2) Forecast / dataset")
    use_mock = st.toggle("Usa Mock Data (Open-Meteo Ensemble like)", value=True)
    n_days_eval = st.slider("Giorni D0 da valutare (a partire dal primo giorno disponibile)", 3, 14, 7)
    risk_aversion = st.slider("Risk aversion (peso dell'incertezza)", 0.0, 2.0, 0.7, 0.1)

    st.caption("Suggerimento: risk_aversion ↑ ⇒ preferisci date con spread P90–P10 più stretto.")

params = CraneParams(
    mob_demob_eur=float(mob_demob),
    op_cost_std_eur_h=float(op_std),
    op_cost_fest_eur_h=float(op_fest),
    standby_cost_eur_h=float(standby),
    shift_start=shift_start,
    shift_end=shift_end,
)

# -----------------------------
# STEPS TABLE
# -----------------------------
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

steps_df = st.data_editor(
    default_steps,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
)

# Build steps list
steps: List[Step] = []
for _, r in steps_df.iterrows():
    name = str(r["Step"])
    dur = float(r["Durata [h]"])
    wt = float(r["Wind Threshold [m/s]"])
    gt = r["Gust Threshold [m/s] (opzionale)"]
    gt = None if (gt is None or (isinstance(gt, float) and np.isnan(gt))) else float(gt)
    minw = float(r["Finestra minima consecutiva [h]"])
    steps.append(Step(name=name, duration_h=dur, wind_thr=wt, gust_thr=gt, min_seq_h=minw))

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "timestamp" not in df.columns:
        raise ValueError("CSV deve contenere una colonna 'timestamp'.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(show_spinner=False)
def load_mock(days: int = 10, members: int = 10, include_gusts: bool = True) -> pd.DataFrame:
    return generate_mock_open_meteo_ensemble(days=days, n_members=members, include_gusts=include_gusts)

if use_mock:
    df = load_mock(days=max(10, n_days_eval + 4), members=10, include_gusts=True)
    st.info("Dataset: Mock Open‑Meteo Ensemble (10 membri) con gusts opzionali inclusi.")
else:
    up = st.file_uploader("Carica CSV forecast orario (timestamp, price_eur_mwh, wind_speed_80m_member_0..N, opzionale gusts)", type=["csv"])
    if up is None:
        st.warning("Carica un CSV oppure attiva 'Usa Mock Data'.")
        st.stop()
    df = load_dataset_from_csv(up)

# Basic checks
required_cols = {"timestamp"}
missing_req = required_cols - set(df.columns)
if missing_req:
    st.error(f"Colonne mancanti: {missing_req}")
    st.stop()

if "price_eur_mwh" not in df.columns:
    st.warning("Colonna 'price_eur_mwh' non trovata: la mancata produzione verrà calcolata come 0€.")
    df["price_eur_mwh"] = 0.0

wind_cols, gust_cols = detect_members(df)
if not wind_cols:
    st.error("Non ho trovato colonne wind ensemble (es. 'wind_speed_80m_member_0').")
    st.stop()

# Show dataset snapshot
with st.expander("Anteprima dataset meteo", expanded=False):
    st.dataframe(df.head(24), use_container_width=True)

# -----------------------------
# RUN SIMULATION FOR MULTIPLE D0
# -----------------------------
st.subheader("Risultati per giorno di inizio (D0)")

df = df.sort_values("timestamp").reset_index(drop=True)
df["date"] = df["timestamp"].dt.date
unique_days = pd.Series(df["date"].unique()).sort_values().tolist()

if len(unique_days) < 2:
    st.error("Dataset troppo corto: servono almeno 2 giorni orari.")
    st.stop()

days_to_test = unique_days[: min(len(unique_days), n_days_eval)]

@st.cache_data(show_spinner=False)
def run_all(days_to_test, df, wind_cols, gust_cols, steps, params):
    sims = {}
    for d in days_to_test:
        sim = simulate_single_start_day(
            df=df,
            start_day=pd.Timestamp(d),
            wind_cols=wind_cols,
            gust_cols=gust_cols,
            steps=steps,
            params=params,
            rated_mw=2.0,
        )
        sims[pd.Timestamp(d)] = sim
    return sims

with st.spinner("Simulazione stocastica in corso (ensemble su più D0)..."):
    sims = run_all(days_to_test, df, wind_cols, gust_cols, steps, params)

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

# Table
display_cols = ["Giorno Inizio (D0)", "Costo Min (P10) €", "Costo Medio Atteso €", "Costo Max (P90) €", "Livello di Confidenza", "Scenari incompleti"]
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

# -----------------------------
# CHARTS
# -----------------------------
c1, c2 = st.columns([1.15, 1.0])
with c1:
    st.plotly_chart(plot_costs_band(summary), use_container_width=True)

with c2:
    st.markdown("### Dettaglio per un D0 (Gantt medio + perdite)")
    selected = st.selectbox(
        "Seleziona giorno D0",
        options=[pd.Timestamp(d).date() for d in days_to_test],
        index=0,
    )
    sim_sel = sims.get(pd.Timestamp(selected))
    if sim_sel is None or sim_sel.get("status") != "ok":
        st.warning("Simulazione non disponibile per il giorno selezionato.")
    else:
        frac = aggregate_gantt(sim_sel["member_logs"])
        if frac.empty:
            st.warning("Nessun log disponibile (dataset troppo corto?).")
        else:
            st.plotly_chart(plot_gantt_fraction(frac), use_container_width=True)

        prod_daily = compute_prod_loss_daily_for_selected(sim_sel)
        if not prod_daily.empty:
            st.plotly_chart(plot_daily_prod_loss_band(prod_daily), use_container_width=True)
        else:
            st.info("Non disponibile la perdita giornaliera (nessun log).")

# -----------------------------
# NOTES / EXPORT HINTS
# -----------------------------
with st.expander("Note tecniche (assunzioni & come adattare a dati reali)", expanded=False):
    st.markdown(
        """
**Assunzioni implementate (coerenti con la tua specifica):**
- Simulazione **oraria**.
- Fuori turno: attività ferma, **costo gru = 0€**, ma la **mancata produzione continua H24**.
- In turno:
  - Se lo step non è partito: si richiede una finestra minima consecutiva di ore “buone” **a partire dall’ora corrente** (approccio dinamico, evita di “buttare via” il turno).
  - Se lo step è già partito: con meteo sopra soglia ⇒ **Standby** (costo standby in turno).
- Festivi: per semplicità qui uso **weekend** (Sab/Dom) come “festivo”.

**Dati Open‑Meteo Ensemble:**
- La struttura mock usa colonne per-membro tipo `wind_speed_80m_member_0..N` e (opzionale) `wind_gusts_10m_member_0..N`.  
  Nella documentazione dell’**Ensemble API** sono disponibili variabili orarie incluse **Wind Speed (80 m)** e **Wind Gusts (10 m)**.  
  Se nel tuo dataset i gusts non sono presenti, il tool ignora automaticamente la soglia raffiche.  

**Power curve:**
- Curva generica 2 MW (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s).  
  Puoi sostituirla con la tua curva reale (tabellare/interpolata) senza cambiare il resto.

**Prezzi energia:**
- Nel mock uso 60–120 €/MWh per dare una forma realistica oraria.  
  Per usare i dati reali, carica un CSV con `timestamp` e `price_eur_mwh` (ora per ora).  
  Il GME consente export dei dati del PUN Index anche con dettaglio intraday.  
        """
    )

st.caption("✅ Pronto: salva come app.py e avvia con `streamlit run app.py`.")

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
# OPEN-METEO: FETCH ENSEMBLE
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
    hourly_vars = ["wind_speed_80m", "wind_speed_100m", "wind_speed_10m"]
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
        wind_keys = _extract_member_cols(hourly, "wind_speed_100m")
        
    if not wind_keys:
        wind_keys = _extract_member_cols(hourly, "wind_speed_10m")

    if not wind_keys:
        chiavi_alternative = [k for k in hourly.keys() if "wind_speed" in k]
        for ca in chiavi_alternative:
            radice = ca.split("_member")[0]
            wind_keys = _extract_member_cols(hourly, radice)
            if wind_keys:
                break

    if not wind_keys:
        raise ValueError(
            f"Impossibile mappare membri ensemble di tipo 'wind_speed' per il modello '{model}'."
        )

    for i, k in enumerate(wind_keys):
        df[f"wind_speed_80m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    gust_keys = _extract_member_cols(hourly, "wind_gusts_10m") if include_gusts else []
    if include_gusts and not gust_keys:
        chiavi_gust = [k for k in hourly.keys() if "wind_gusts" in k]
        for cg in chiavi_gust:
            radice = cg.split("_member")[0]
            gust_keys = _extract_member_cols(hourly, radice)
            if gust_keys:
                break

    if include_gusts and gust_keys:
        for i, k in enumerate(gust_keys):
            if i < len(wind_keys):
                df[f"wind_gusts_10m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    df["price_eur_mwh"] = generate_price_series(df["timestamp"])
    return df

# -----------------------------
# MOCK DATA
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
# PLOTS
# -----------------------------
def plot_wind_speed_ensemble(df_view: pd.DataFrame, wind_cols: List[str]) -> go.Figure:
    wind_mat = df_view[wind_cols].to_numpy(dtype=float)
    p10 = np.percentile(wind_mat, 10, axis=1)
    p90 = np.percentile(wind_mat, 90, axis=1)
    mean = np.mean(wind_mat, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view["timestamp"], y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=df_view["timestamp"],
            y=p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(14, 165, 233, 0.18)",
            name="Incertezza Vento (P10–P90)",
        )
    )
    fig.add_trace(go.Scatter(x=df_view["timestamp"], y=mean, mode="lines", line=dict(color="rgba(14, 165, 233, 1)", width=2.8), name="Vento Medio"))
    fig.update_layout(title="Velocità del vento prevista [m/s]", template="plotly_white", height=320)
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
    requires_crane: bool

def check_min_window(df: pd.DataFrame, start_idx: int, wind_col: str, gust_col: Optional[str], step: Step, params: CraneParams) -> bool:
    needed = int(math.ceil(max(0.0, step.min_seq_h)))
    if needed <= 1:
        return True
    ts0 = df.at[start_idx, "timestamp"]
    shift_end_ts = shift_end_timestamp(ts0, params.shift_end)

    for k in range(needed):
        idx = start_idx + k
        if idx >= len(df): return False
        ts = df.at[idx, "timestamp"]
        if ts >= shift_end_ts: return False
        if not in_work_shift(ts, params.shift_start, params.shift_end): return False
        w = df.at[idx, wind_col]
        if pd.isna(w): return False
        ok = float(w) < float(step.wind_thr)
        if gust_col is not None and step.gust_thr is not None:
            g = df.at[idx, gust_col]
            if pd.isna(g): return False
            ok = ok and (float(g) < float(step.gust_thr))
        if not ok: return False
    return True

def op_cost_for_hour(ts: pd.Timestamp, params: CraneParams) -> float:
    return params.op_cost_fest_eur_h if is_weekend(ts) else params.op_cost_std_eur_h

def simulate_single_start_day_profit(
    df: pd.DataFrame, horizon_start: pd.Timestamp, start_day: pd.Timestamp,
    wind_cols: List[str], gust_cols: Optional[List[str]], steps: List[Step],
    params: CraneParams, rated_mw: float = 2.0
) -> Dict:
    df = df.sort_values("timestamp").reset_index(drop=True)
    start_ts = pd.Timestamp(start_day.date())
    start_idx = int(df["timestamp"].searchsorted(start_ts))
    if start_idx >= len(df):
        return {"status": "out_of_range"}

    member_rows = []
    member_logs = []
    mob_demob_apply = params.mob_demob_eur if any(s.requires_crane for s in steps) else 0.0

    for m, wind_col in enumerate(wind_cols):
        gust_col = gust_cols[m] if gust_cols is not None else None
        step_i = 0
        remaining = float(steps[0].duration_h) if steps else 0.0
        current_step_started = False

        crane_cost = 0.0
        lost_revenue = 0.0
        logs = []
        idx = start_idx
        last_ts = df["timestamp"].iloc[min(start_idx, len(df)-1)]

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
                    if crane_present: c_cost = op_cost_for_hour(ts, params)
                else:
                    state = "Standby" if crane_present else "Attesa (no gru)"
                    if crane_present: c_cost = params.standby_cost_eur_h

                crane_cost += c_cost
                if remaining <= 1e-9:
                    step_i += 1
                    if step_i < len(steps):
                        remaining = float(steps[step_i].duration_h)
                        current_step_started = False

            logs.append({
                "timestamp": ts, "state": state, "crane_cost_eur": c_cost,
                "prod_loss_eur": loss_eur, "step_name": step.name, "member": m, "crane_present": crane_present
            })
            idx += 1

        partial = (step_i < len(steps))
        costo_intervento = mob_demob_apply + crane_cost + lost_revenue

        member_rows.append({
            "member": m, "profit_net_eur": costo_intervento,
            "partial": partial, "completion_ts": last_ts
        })
        member_logs.append(pd.DataFrame(logs))

    return {
        "status": "ok", "start_day": start_ts,
        "member_results": pd.DataFrame(member_rows), "member_logs": member_logs
    }

# -----------------------------
# PROBABILISTIC LOGIC (SUMMARY)
# -----------------------------
def compute_daily_summary_profit(all_sims: Dict[pd.Timestamp, Dict]) -> pd.DataFrame:
    rows = []
    for d0, sim in all_sims.items():
        if sim.get("status") != "ok": continue
        mr = sim["member_results"]
        
        # LOGICA RICHIESTA: Estrae puramente la probabilità reale di successo per tutta l'attività
        total_scenarios = len(mr)
        successful_scenarios = len(mr[mr["partial"] == False])
        prob_success = (successful_scenarios / total_scenarios) * 100.0 if total_scenarios > 0 else 0.0

        costi = mr["profit_net_eur"].to_numpy(dtype=float)
        p10 = safe_percentile(costi, 10)
        p90 = safe_percentile(costi, 90)
        mean = float(np.nanmean(costi[np.isfinite(costi)])) if np.any(np.isfinite(costi)) else np.nan
        spread = p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan

        rows.append({
            "Giorno Inizio (D0)": pd.Timestamp(d0).date(),
            "Probabilità Successo (%)": prob_success,
            "Perdita P10 €": p10,
            "Perdita Media €": mean,
            "Perdita P90 €": p90,
            "Spread (P90-P10) €": spread
        })

    return pd.DataFrame(rows).sort_values("Giorno Inizio (D0)").reset_index(drop=True)

def choose_optimal_day_profit(summary: pd.DataFrame, risk_aversion: float = 0.7) -> Tuple[Optional[pd.Timestamp], pd.DataFrame]:
    s = summary.copy()
    if s.empty: return None, s

    mean = s["Perdita Media €"].to_numpy(dtype=float)
    spread = s["Spread (P90-P10) €"].to_numpy(dtype=float)

    # Formula dello score lineare e pulito (costo + avversione al rischio dello spread finanziario)
    score = mean + float(risk_aversion) * np.nan_to_num(spread, nan=0.0)
    s["Score (min meglio)"] = score

    best_idx = int(np.nanargmin(score))
    best_day = pd.Timestamp(s.loc[best_idx, "Giorno Inizio (D0)"])
    return best_day, s

def heuristic_estimate_seconds(n_members: int, n_days: int, n_steps: int) -> float:
    return max(0.5, n_members * n_days * n_steps * 0.0018)

def format_time_estimate(seconds: float) -> str:
    if seconds < 1: return "< 1 secondo"
    if seconds < 60: return f"~{seconds:.0f} secondi"
    return f"~{seconds/60:.1f} minuti"

# -----------------------------
# PLOT CANDLES
# -----------------------------
def plot_profit_candles(summary_scored: pd.DataFrame) -> go.Figure:
    if summary_scored.empty: return go.Figure()
    dfp = summary_scored.copy().sort_values("Giorno Inizio (D0)")
    x = dfp["Giorno Inizio (D0)"].astype(str)
    
    opens, closes = [], []
    for i in range(len(dfp)):
        current_mean = dfp.iloc[i]["Perdita Media €"]
        prev_mean = dfp.iloc[i-1]["Perdita Media €"] if i > 0 else current_mean
        spessore = current_mean * 0.015
        if current_mean <= prev_mean:
            opens.append(current_mean + spessore)
            closes.append(current_mean - spessore)
        else:
            opens.append(current_mean - spessore)
            closes.append(current_mean + spessore)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=opens, high=dfp["Perdita P90 €"], low=dfp["Perdita P10 €"], close=closes,
        name="Impatto Finanziario",
        hovertemplate="<b>Data D0: %{x}</b><br>Max (P90): %{high:,.0f} €<br>Media: %{close:,.0f} €<br>Min (P10): %{low:,.0f} €<extra></extra>"
    ))
    fig.update_traces(increasing=dict(fillcolor="#f59e0b", line=dict(color="#f59e0b")), decreasing=dict(fillcolor="#06b6d4", line=dict(color="#06b6d4")), whiskerwidth=0)
    fig.update_layout(title="Perdita Totale Intervento vs Giorno D0 (Inclusi rischi meteo)", yaxis_title="Euro (€)", template="plotly_white", height=420, xaxis=dict(rangeslider=dict(visible=False)))
    return fig

# -----------------------------
# GANTT DETAILED
# -----------------------------
def aggregate_gantt(member_logs: List[pd.DataFrame]) -> pd.DataFrame:
    if not member_logs: return pd.DataFrame()
    all_logs = pd.concat(member_logs, ignore_index=True)
    n_members = all_logs["member"].nunique()
    pivot = all_logs.groupby(["timestamp", "state"])["member"].nunique().unstack(fill_value=0).sort_index()
    for col in ["Lavoro", "Standby", "Stop Notte", "Attesa (no gru)"]:
        if col not in pivot.columns: pivot[col] = 0
    frac = (pivot / max(1, n_members)).reset_index()
    return frac

def plot_gantt_fraction(frac: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "Lavoro" in frac.columns: fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Lavoro"], name="Lavoro", marker_color="rgba(34,197,94,0.85)"))
    if "Standby" in frac.columns: fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Standby"], name="Standby", marker_color="rgba(245,158,11,0.85)"))
    if "Attesa (no gru)" in frac.columns: fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Attesa (no gru)"], name="Attesa (no gru)", marker_color="rgba(59,130,246,0.3)"))
    if "Stop Notte" in frac.columns: fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Stop Notte"], name="Stop Notte", marker_color="rgba(148,163,184,0.85)"))
    fig.update_layout(barmode="stack", title="Gantt Medio (Quota scenari per stato orario)", template="plotly_white", height=380)
    return fig

# -----------------------------
# UI STREAMLIT APP
# -----------------------------
st.title("WTG Main Component – Loss Minimizer (Stochastic Feasibility)")

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
    forecast_days = st.slider("Giorni forecast", 3, 16, 10)
    include_gusts = st.toggle("Usa raffiche", value=True)
    st.divider()
    st.header("C) Campionamento")
    use_all_members = st.toggle("Usa tutti i membri", value=False)
    # CORREZIONE ERRORE RIGA 623: 'St' modificato nel corretto alias di streamlit 'st'
    n_members_input = st.number_input("Membri da usare", min_value=1, value=10, disabled=use_all_members)
    st.divider()
    st.header("D) Ottimizzazione")
    earliest_day = st.date_input("Data di inizio minima", value=pd.Timestamp.now().date())
    risk_aversion = st.slider("Risk aversion", 0.0, 3.0, 0.7)
    use_mock = st.toggle("Mock Data (debug)", value=False)

# Build Steps Table
default_steps = pd.DataFrame({
    "Step": ["Step 1", "Step 2", "Step 3"], "Durata [h]": [8.0, 8.0, 8.0],
    "Wind Threshold [m/s]": [8.0, 8.0, 8.0], "Gust Threshold [m/s] (opzionale)": [np.nan, np.nan, np.nan],
    "Finestra minima consecutiva [h]": [3.0, 3.0, 3.0], "Richiede Gru": [True, True, True],
})
st.subheader("Pianificazione Attività")
steps_df = st.data_editor(default_steps, num_rows="dynamic", use_container_width=True, hide_index=True)

steps = []
for _, r in steps_df.iterrows():
    name = str(r.get("Step", "")).strip()
    dur = safe_float(r.get("Durata [h]"))
    wt = safe_float(r.get("Wind Threshold [m/s]"))
    minw = safe_float(r.get("Finestra minima consecutiva [h]"), default=0.0)
    gt_raw = r.get("Gust Threshold [m/s] (opzionale)")
    gt = None if (gt_raw is None or pd.isna(gt_raw)) else safe_float(gt_raw)
    req = safe_bool(r.get("Richiede Gru"))
    if dur and wt:
        steps.append(Step(name=name if name else "Step", duration_h=dur, wind_thr=wt, gust_thr=gt, min_seq_h=minw, requires_crane=req))

if not steps:
    st.error("Inserisci almeno uno step valido.")
    st.stop()

# Load Data
with st.spinner("Caricamento dati meteo..."):
    if use_mock:
        df = generate_mock_open_meteo_ensemble(days=int(forecast_days), n_members=20, include_gusts=include_gusts)
    else:
        df = fetch_open_meteo_ensemble(latitude, longitude, model, forecast_days, include_gusts)

df["timestamp"] = pd.to_datetime(df["timestamp"])
wind_cols_all, gust_cols_all = detect_members(df)
wind_cols_use = wind_cols_all if use_all_members else wind_cols_all[:min(len(wind_cols_all), int(n_members_input))]
gust_cols_use = gust_cols_all[:len(wind_cols_use)] if gust_cols_all else None

st.plotly_chart(plot_wind_speed_ensemble(df, wind_cols_use), use_container_width=True)

# Run Simulation
all_days = pd.date_range(start=max(df["timestamp"].min().normalize(), pd.Timestamp(earliest_day)), end=df["timestamp"].max().normalize(), freq="1D")
est_sec = heuristic_estimate_seconds(len(wind_cols_use), len(all_days), len(steps))

st.info(f"⏱️ **Tempo stimato di calcolo:** {format_time_estimate(est_sec)} ({len(wind_cols_use)} membri × {len(all_days)} giorni)")

cur_hash = hashlib.md5(json.dumps({"m": len(wind_cols_use), "d": [str(x) for x in all_days], "ra": risk_aversion}, sort_keys=True).encode()).hexdigest()
if "sims_profit" not in st.session_state: st.session_state["sims_profit"] = None
if "sims_hash_profit" not in st.session_state: st.session_state["sims_hash_profit"] = None

if st.button("▶ Esegui simulazione probabilistica", type="primary", use_container_width=True):
    sims = {}
    prog = st.progress(0.0)
    for i, d0 in enumerate(all_days):
        d0_ts = pd.Timestamp(d0)
        sims[d0_ts] = simulate_single_start_day_profit(df, df["timestamp"].min(), d0_ts, wind_cols_use, gust_cols_use, steps, params)
        prog.progress((i + 1) / len(all_days))
    prog.empty()
    st.session_state["sims_profit"] = sims
    st.session_state["sims_hash_profit"] = cur_hash

# Render Results
if st.session_state["sims_profit"]:
    sims = st.session_state["sims_profit"]
    summary = compute_daily_summary_profit(sims)

    if not summary.empty:
        best_d0, scored = choose_optimal_day_profit(summary, risk_aversion=risk_aversion)
        
        st.header("Classifica Ottimizzazione di Finestra Operativa")
        c1, c2, c3 = st.columns(3)
        if best_d0:
            rb = scored[scored["Giorno Inizio (D0)"] == best_d0.date()].iloc[0]
            c1.metric("Miglior Giorno D0 Suggerito", best_d0.strftime("%d/%m/%Y"))
            c2.metric("Probabilità di Successo Completamento", f"{rb['Probabilità Successo (%)']:.1f} %")
            c3.metric("Costo Medio Atteso", f"{rb['Perdita Media €']:,.0f} €")

        st.subheader("Tabella di Fattibilità ed Impatto Economico Completa")
        st.dataframe(
            scored.style.format({
                "Probabilità Successo (%)": "{:.1f} %", "Perdita P10 €": "{:,.2f} €",
                "Perdita Media €": "{:,.2f} €", "Perdita P90 €": "{:,.2f} €",
                "Spread (P90-P10) €": "{:,.2f} €", "Score (min meglio)": "{:,.2f}"
            }).background_gradient(subset=["Probabilità Successo (%)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True
        )

        st.plotly_chart(plot_profit_candles(scored), use_container_width=True)

        st.markdown("### Dettaglio Orario (Gantt stocastico del giorno scelto)")
        selected = st.selectbox("Seleziona D0 per vedere il Gantt medio e le probabilità orarie", scored["Giorno Inizio (D0)"].tolist())
        sim_sel = sims.get(pd.Timestamp(selected))
        if sim_sel:
            frac = aggregate_gantt(sim_sel["member_logs"])
            if not frac.empty:
                st.plotly_chart(plot_gantt_fraction(frac), use_container_width=True)

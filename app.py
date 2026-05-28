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
            hovertemplate="Vento: %{x:.1f} m/s<br>Potenza: %{y:.2f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        title="Power Curve Turbina (2 MW)",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="Power [MW]",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=55, b=10),
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
    # Chiediamo 80m, 100m (fallback ottimo per ECMWF) e 10m per massima robustezza multipiattaforma
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

    # --- CASCATA FALLBACK MEMBRI ENSEMBLE VENTO ---
    wind_keys = _extract_member_cols(hourly, "wind_speed_80m")
    var_usata = "wind_speed_80m"
    
    if not wind_keys:
        # Se mancano gli 80m (tipico di ECMWF IFS), ripieghiamo sui 100m che sono eccellenti per quote hub
        wind_keys = _extract_member_cols(hourly, "wind_speed_100m")
        var_usata = "wind_speed_100m"
        
    if not wind_keys:
        # Estrema ratio sui 10 metri
        wind_keys = _extract_member_cols(hourly, "wind_speed_10m")
        var_usata = "wind_speed_10m"

    if not wind_keys:
        # Scansione dinamica disperata su qualsiasi chiave che contenga 'wind_speed'
        chiavi_alternative = [k for k in hourly.keys() if "wind_speed" in k]
        for ca in chiavi_alternative:
            radice = ca.split("_member")[0]
            wind_keys = _extract_member_cols(hourly, radice)
            if wind_keys:
                var_usata = radice
                break

    if not wind_keys:
        raise ValueError(
            f"Impossibile mappare membri ensemble di tipo 'wind_speed' per il modello '{model}'. "
            f"Variabili ritornate dall'API: {list(hourly.keys())}"
        )

    # Standardizziamo le colonne iniettate nel DF con il nome atteso dal codice di simulazione
    for i, k in enumerate(wind_keys):
        df[f"wind_speed_80m_member_{i}"] = pd.to_numeric(hourly[k], errors="coerce")

    # --- CASCATA FALLBACK MEMBRI ENSEMBLE RAFFICHE ---
    gust_keys = _extract_member_cols(hourly, "wind_gusts_10m") if include_gusts else []
    if include_gusts and not gust_keys:
        # Se mancano le raffiche a 10m, verifichiamo se esistono ad altre quote
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
# PLOTS: METEO & EXPECTED PRODUCTION
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
            hovertemplate="%{x|%d/%m %H:%M}<br>Prezzo: %{y:.2f} €/MWh<extra></extra>",
        )
    )
    fig.update_layout(
        title="Prezzi orari energia [€/MWh]",
        xaxis_title="Tempo",
        yaxis_title="€/MWh",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    return fig

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
            hovertemplate="%{x|%d/%m %H:%M}<br>P10: %{y:.2f} m/s<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_view["timestamp"],
            y=mean,
            mode="lines",
            line=dict(color="rgba(14, 165, 233, 1)", width=2.8),
            name="Velocità vento media",
            hovertemplate="%{x|%d/%m %H:%M}<br>Media: %{y:.2f} m/s<extra></extra>",
        )
    )
    fig.update_layout(
        title="Velocità del vento prevista [m/s] (media ensemble + banda P10–P90)",
        xaxis_title="Tempo",
        yaxis_title="Vento [m/s]",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    return fig

def plot_expected_production(df_view: pd.DataFrame, wind_cols: List[str]) -> go.Figure:
    wind_mat = df_view[wind_cols].to_numpy(dtype=float)
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
            name="P10–P90",
            hovertemplate="%{x|%d/%m %H:%M}<br>P10: %{y:.2f} MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_view["timestamp"],
            y=mean,
            mode="lines",
            line=dict(color="rgba(59,130,246,1)", width=2.4),
            name="Media",
            hovertemplate="%{x|%d/%m %H:%M}<br>Media: %{y:.2f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        title="Produzione prevista oraria [MW] (media ensemble + banda P10–P90)",
        xaxis_title="Tempo",
        yaxis_title="MW",
        template="plotly_white",
        height=320,
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
    requires_crane: bool

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

def shift_hours_length(params: CraneParams) -> float:
    t0 = pd.to_datetime(params.shift_start, format="%H:%M")
    t1 = pd.to_datetime(params.shift_end, format="%H:%M")
    return max(1.0, (t1 - t0).total_seconds() / 3600.0)

def simulate_single_start_day_profit(
    df: pd.DataFrame,
    horizon_start: pd.Timestamp,
    start_day: pd.Timestamp,
    wind_cols: List[str],
    gust_cols: Optional[List[str]],
    steps: List[Step],
    params: CraneParams,
    rated_mw: float = 2.0,
) -> Dict:
    df = df.sort_values("timestamp").reset_index(drop=True)

    start_ts = pd.Timestamp(start_day.date())
    start_idx = int(df["timestamp"].searchsorted(start_ts))
    if start_idx >= len(df):
        return {"status": "out_of_range"}

    member_rows = []
    member_logs = []

    mob_demob_apply = params.mob_demob_eur if any(s.requires_crane for s in steps) else 0.0
    shift_len = shift_hours_length(params)

    for m, wind_col in enumerate(wind_cols):
        gust_col = gust_cols[m] if gust_cols is not None else None

        step_i = 0
        remaining = float(steps[0].duration_h) if steps else 0.0
        current_step_started = False

        crane_cost = 0.0
        lost_revenue = 0.0
        fermo_hours_sim = 0
        crane_shift_hours_charged = 0

        logs = []
        idx = start_idx
        last_ts = df["timestamp"].iloc[min(start_idx, len(df)-1)]

        while idx < len(df) and step_i < len(steps):
            ts = df.at[idx, "timestamp"]
            last_ts = ts

            step = steps[step_i]
            step_name_for_log = step.name

            w = df.at[idx, wind_col]
            g = df.at[idx, gust_col] if gust_col is not None else np.nan
            p_mw = power_curve_mw(w, rated_mw=rated_mw)
            price = float(df.at[idx, "price_eur_mwh"])
            loss_eur = p_mw * price
            lost_revenue += loss_eur
            fermo_hours_sim += 1

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
                        crane_shift_hours_charged += 1
                else:
                    state = "Standby" if crane_present else "Attesa (no gru)"
                    if crane_present:
                        c_cost = params.standby_cost_eur_h
                        crane_shift_hours_charged += 1

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
                    "crane_present": crane_present,
                }
            )
            idx += 1

        if step_i < len(steps):
            partial = True
            remaining_work_h = max(0.0, remaining)
            if step_i + 1 < len(steps):
                remaining_work_h += float(sum(s.duration_h for s in steps[step_i+1:]))
        else:
            partial = False
            remaining_work_h = 0.0

        if fermo_hours_sim > 0:
            avg_loss_per_hour = lost_revenue / fermo_hours_sim
        else:
            avg_loss_per_hour = 0.0

        if crane_shift_hours_charged > 0:
            avg_crane_per_shift_hour = crane_cost / crane_shift_hours_charged
        else:
            avg_crane_per_shift_hour = 0.0

        calendar_hours_remaining = (remaining_work_h * 24.0 / shift_len) if shift_len > 0 else remaining_work_h
        penalty_eur = 0.0
        if partial and remaining_work_h > 0:
            penalty_eur = (avg_loss_per_hour * calendar_hours_remaining) + (avg_crane_per_shift_hour * remaining_work_h)

        # STRATEGIA 2: Perdita Totale dell'Intervento (valore positivo da minimizzare)
        costo_totale_intervento = mob_demob_apply + crane_cost + lost_revenue

        member_rows.append(
            {
                "member": m,
                "profit_net_eur": costo_totale_intervento,
                "mob_demob_eur": mob_demob_apply,
                "crane_cost_eur": crane_cost,
                "lost_revenue_eur": lost_revenue,
                "completion_ts": last_ts,
                "partial": partial,
                "remaining_work_h": remaining_work_h,
                "penalty_eur": penalty_eur,
                "fermo_hours_sim": fermo_hours_sim,
                "crane_shift_hours_charged": crane_shift_hours_charged,
            }
        )
        member_logs.append(pd.DataFrame(logs))

    return {
        "status": "ok",
        "start_day": start_ts,
        "member_results": pd.DataFrame(member_rows),
        "member_logs": member_logs,
    }

def compute_daily_summary_profit(all_sims: Dict[pd.Timestamp, Dict]) -> pd.DataFrame:
    rows = []
    for d0, sim in all_sims.items():
        if sim.get("status") != "ok":
            continue
        mr = sim["member_results"]

        # --- FILTRO FLESSIBILE ---
        # Calcola la frazione di membri rimasti a metà (lavoro parziale)
        quota_incompleti = mr["partial"].mean() 
        
        # Se più del 20% dei modelli stocastici prevede un fallimento/ritardo imprevisto, 
        # consideriamo la giornata non pianificabile operativamente.
        if quota_incompleti > 0.20: 
            continue

        costi = mr["profit_net_eur"].to_numpy(dtype=float)
        p10 = safe_percentile(costi, 10)
        p90 = safe_percentile(costi, 90)
        mean = float(np.nanmean(costi[np.isfinite(costi)])) if np.any(np.isfinite(costi)) else np.nan
        spread = p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan

        rows.append(
            {
                "Giorno Inizio (D0)": pd.Timestamp(d0).date(),
                "Perdita P10 €": p10,
                "Perdita Media €": mean,
                "Perdita P90 €": p90,
                "Spread (P90-P10) €": spread,
            }
        )

    # Gestione di emergenza se nessuna giornata supera il filtro impostato
    if not rows:
        return pd.DataFrame(columns=[
            "Giorno Inizio (D0)", 
            "Perdita P10 €", 
            "Perdita Media €", 
            "Perdita P90 €", 
            "Spread (P90-P10) €"
        ])

    # Creazione del DataFrame finale ordinato cronologicamente
    out = pd.DataFrame(rows).sort_values("Giorno Inizio (D0)")
    return out.reset_index(drop=True)

    # Gestione del caso in cui nessuna giornata superi il filtro
    if not rows:
        return pd.DataFrame(columns=[
            "Giorno Inizio (D0)", 
            "Perdita P10 €", 
            "Perdita Media €", 
            "Perdita P90 €", 
            "Spread (P90-P10) €"
        ])

    out = pd.DataFrame(rows).sort_values("Giorno Inizio (D0)")
    return out.reset_index(drop=True)

        # FILTRO FLASSIBILE: Accettiamo il giorno se almeno il 80% degli scenari completa il lavoro quota_incompleti = mr["partial"].mean() # Calcola la frazione di membri rimasti a metà
    if quota_incompleti > 0.20: # Se più del 10% dei modelli prevede un fallimento, scarta il giorno
           continue

        costi = mr["profit_net_eur"].to_numpy(dtype=float)
        p10 = safe_percentile(costi, 10)
        p90 = safe_percentile(costi, 90)
        mean = float(np.nanmean(costi[np.isfinite(costi)])) if np.any(np.isfinite(costi)) else np.nan
        spread = p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan

        rows.append(
            {
                "Giorno Inizio (D0)": pd.Timestamp(d0).date(),
                "Perdita P10 €": p10,
                "Perdita Media €": mean,
                "Perdita P90 €": p90,
                "Spread (P90-P10) €": spread,
            }
        )

    out = pd.DataFrame(rows).sort_values("Giorno Inizio (D0)")
    return out.reset_index(drop=True)

def choose_optimal_day_profit(summary: pd.DataFrame, risk_aversion: float = 0.7) -> Tuple[Optional[pd.Timestamp], pd.DataFrame]:
    s = summary.copy()
    if s.empty:
        return None, s

    mean = s["Perdita Media €"].to_numpy(dtype=float)
    spread = s["Spread (P90-P10) €"].to_numpy(dtype=float)

    score = mean + float(risk_aversion) * np.nan_to_num(spread, nan=0.0)
    s["Score (min meglio)"] = score

    if not np.any(np.isfinite(score)):
        return None, s

    best_idx = int(np.nanargmin(score))  # Cerchiamo il MINIMO costo combinato
    best_day = pd.Timestamp(s.loc[best_idx, "Giorno Inizio (D0)"])
    return best_day, s

# -----------------------------
# FINANCIAL CANDLESTICK PLOT
# -----------------------------
def plot_profit_candles(summary_scored: pd.DataFrame) -> go.Figure:
    if summary_scored.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Nessun giorno D0 disponibile con completamento garantito al 100%",
            template="plotly_white"
        )
        return fig

    dfp = summary_scored.copy().sort_values("Giorno Inizio (D0)")
    x = dfp["Giorno Inizio (D0)"].astype(str)
    
    opens = []
    closes = []

    for i in range(len(dfp)):
        current_mean = dfp.iloc[i]["Perdita Media €"]
        prev_mean = dfp.iloc[i-1]["Perdita Media €"] if i > 0 else current_mean
        
        spessore = current_mean * 0.03
        if current_mean <= prev_mean:
            # Ribasso del costo rispetto a ieri -> Azzurro
            opens.append(current_mean + spessore)
            closes.append(current_mean - spessore)
        else:
            # Rialzo del costo rispetto a ieri -> Arancione
            opens.append(current_mean - spessore)
            closes.append(current_mean + spessore)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=opens,
            high=dfp["Perdita P90 €"],
            low=dfp["Perdita P10 €"],
            close=closes,
            name="Impatto Finanziario",
            hoverinfo="x+y",
            hovertemplate=(
                "<b>Data D0: %{x}</b><br>"
                "Max Rischio (P90): %{high:,.0f} €<br>"
                "Costo Medio Atteso: %{close:,.0f} €<br>"
                "Min Rischio (P10): %{low:,.0f} €<br><extra></extra>"
            )
        )
    )

    # Stile esatto image_af30c0.png
    fig.update_traces(
        increasing=dict(fillcolor="#f59e0b", line=dict(color="#f59e0b", width=1)),
        decreasing=dict(fillcolor="#06b6d4", line=dict(color="#06b6d4", width=1)),
        whiskerwidth=0,
        line=dict(width=1.5)
    )

    fig.update_layout(
        title="Perdita Totale dell'Intervento vs Giorno di Inizio D0 (Solo Giorni Completati al 100%)",
        yaxis_title="Perdita Totale (€)",
        xaxis_title="Giorno di inizio dell'attività (D0)",
        template="plotly_white",
        height=450,
        margin=dict(l=15, r=15, t=60, b=20),
        xaxis=dict(
            type="category",
            rangeslider=dict(visible=False),
            gridcolor="rgba(255,255,255,1)",
        ),
        yaxis=dict(
            gridcolor="rgba(241,245,249,1)",
            zeroline=False
        ),
        plot_bgcolor="#f8fafc",
    )
    return fig

# -----------------------------
# GANTT & LOSS DETAILED PLOTS
# -----------------------------
def aggregate_gantt(member_logs: List[pd.DataFrame]) -> pd.DataFrame:
    if not member_logs:
        return pd.DataFrame()

    all_logs = pd.concat(member_logs, ignore_index=True)
    all_logs = all_logs[["timestamp", "state", "member"]]
    n_members = all_logs["member"].nunique()

    pivot = all_logs.groupby(["timestamp", "state"])["member"].nunique().unstack(fill_value=0).sort_index()
    for col in ["Lavoro", "Standby", "Stop Notte", "Attesa (no gru)"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[[c for c in ["Lavoro", "Standby", "Attesa (no gru)", "Stop Notte"] if c in pivot.columns]]
    frac = (pivot / max(1, n_members)).reset_index()
    return frac

def plot_gantt_fraction(frac: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "Lavoro" in frac.columns:
        fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Lavoro"], name="Lavoro", marker_color="rgba(34,197,94,0.85)"))
    if "Standby" in frac.columns:
        fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Standby"], name="Standby", marker_color="rgba(245,158,11,0.85)"))
    if "Attesa (no gru)" in frac.columns:
        fig.add_trace(go.Bar(x=frac["timestamp"], y=frac["Attesa (no gru)"], name="Attesa (no gru)", marker_color="rgba(59,130,246,0.30)"))
    if "Stop Notte" in frac.columns:
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
        hovermode="x unified",
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
            name="P10–P90",
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

# -----------------------------
# TIME ESTIMATE
# -----------------------------
def format_time_estimate(seconds: float) -> str:
    if seconds <= 0:
        return "n.d."
    if seconds < 1:
        return "< 1 secondo"
    if seconds < 60:
        return f"~{seconds:.0f} secondi"
    return f"~{seconds/60:.1f} minuti"

def heuristic_estimate_seconds(n_members: int, n_days: int, horizon_hours: int, total_work_h: float) -> float:
    if n_members <= 0 or n_days <= 0:
        return 0.0
    avg_hours = max(1.0, horizon_hours / 2.0)
    ops = n_members * n_days * (avg_hours + 2.0 * total_work_h)
    return ops / 45000.0

# -----------------------------
# UI STREAMLIT APP
# -----------------------------
st.title("WTG Main Component – Loss Minimizer (Open‑Meteo Ensemble)")

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
        options=["gfs_seamless", "icon_seamless", "ecmwf_ifs04"],
        index=0,
    )
    forecast_days = st.slider("Forecast days (max 16)", min_value=3, max_value=16, value=10, step=1)
    include_gusts = st.toggle("Usa wind_gusts_10m (se disponibili)", value=True)
    st.divider()
    st.header("C) Simulazione ensemble")
    use_all_members = st.toggle("Usa tutti i membri disponibili", value=False)
    n_members_input = st.number_input(
        "Numero di membri ensemble da usare (se non full)",
        min_value=1, max_value=80, value=10, step=1,
        disabled=use_all_members
    )
    st.divider()
    st.header("D) Pianificazione e rischio")
    earliest_day = st.date_input("Primo giorno organizzabile (earliest D0)", value=pd.Timestamp.now().date())
    risk_aversion = st.slider("Risk aversion (penalità spread)", 0.0, 3.0, 0.7, 0.1)
    st.divider()
    st.header("E) Debug")
    use_mock = st.toggle("Usa Mock Data (solo debug)", value=False)

# Validazione turni
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

# Tabella attività modificabile
st.subheader("Attività (step sequenziali)")
default_steps = pd.DataFrame(
    {
        "Step": ["Step 1", "Step 2", "Step 3"],
        "Durata [h]": [8.0, 8.0, 8.0],
        "Wind Threshold [m/s]": [8.0, 8.0, 8.0],
        "Gust Threshold [m/s] (opzionale)": [np.nan, np.nan, np.nan],
        "Finestra minima consecutiva [h]": [3.0, 3.0, 3.0],
        "Richiede Gru": [True, True, True],
    }
)
steps_df = st.data_editor(default_steps, num_rows="dynamic", use_container_width=True, hide_index=True)

steps: List[Step] = []
for _, r in steps_df.iterrows():
    name = str(r.get("Step", "")).strip()
    dur = safe_float(r.get("Durata [h]"), default=None)
    wt = safe_float(r.get("Wind Threshold [m/s]"), default=None)
    minw = safe_float(r.get("Finestra minima consecutiva [h]"), default=0.0)
    gt_raw = r.get("Gust Threshold [m/s] (opzionale)")
    gt = None if (gt_raw is None or (isinstance(gt_raw, float) and np.isnan(gt_raw))) else safe_float(gt_raw, default=None)
    req = safe_bool(r.get("Richiede Gru"), default=True)
    
    if name == "" and dur is None and wt is None:
        continue
    if dur is None or dur <= 0 or wt is None or wt <= 0:
        st.warning("⚠️ Una o più righe step hanno durata/soglia non valida. Correggi per includerle.")
        continue
    steps.append(Step(name=name if name != "" else "Step", duration_h=float(dur), wind_thr=float(wt), gust_thr=gt, min_seq_h=float(minw or 0), requires_crane=req))

if len(steps) == 0:
    st.error("Nessuno step valido definito. Inserisci almeno uno step con Durata > 0 e Wind Threshold > 0.")
    st.stop()

total_work_h = float(sum(s.duration_h for s in steps))

# Caricamento meteo sincrono
with st.spinner("Caricamento forecast Open‑Meteo..."):
    if use_mock:
        df = generate_mock_open_meteo_ensemble(days=int(forecast_days), n_members=20, include_gusts=include_gusts)
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
    st.success("Forecast caricato correttamente tramite le API di Open‑Meteo.")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["date"] = df["timestamp"].dt.date
wind_cols_all, gust_cols_all = detect_members(df)

if not wind_cols_all:
    st.error("Non ho trovato colonne wind ensemble nel dataset.")
    st.stop()

n_members_available = len(wind_cols_all)
if use_all_members:
    wind_cols_use = wind_cols_all
    gust_cols_use = gust_cols_all
else:
    n_members_to_use = min(int(n_members_input), n_members_available)
    if n_members_to_use >= n_members_available:
        wind_cols_use = wind_cols_all
        gust_cols_use = gust_cols_all
    else:
        indices = np.linspace(0, n_members_available - 1, n_members_to_use, dtype=int)
        indices = sorted(list(set(indices)))
        wind_cols_use = [wind_cols_all[i] for i in indices]
        gust_cols_use = [gust_cols_all[i] for i in indices] if gust_cols_all is not None else None
        st.info(f"💡 Campionamento Ensemble attivo: estratti {len(wind_cols_use)} membri distribuiti sul fascio.")

forecast_start = df["timestamp"].min()
forecast_end = df["timestamp"].max()
earliest_ts = pd.Timestamp(earliest_day)
horizon_start = max(forecast_start.normalize(), clamp_date_to_forecast(earliest_ts, forecast_start, forecast_end))

st.markdown("### Grafici Meteo & Produzione Prevista")
st.plotly_chart(plot_wind_speed_ensemble(df, wind_cols_use), use_container_width=True)
st.plotly_chart(plot_expected_production(df, wind_cols_use), use_container_width=True)

# --- CONTROLLO ED ESECUZIONE SIMULAZIONE ---
all_days = pd.date_range(start=horizon_start, end=forecast_end.normalize(), freq="1D")

if len(all_days) == 0:
    st.error("Nessun giorno D0 analizzabile nell'orizzonte forecast.")
    st.stop()

total_sims = len(all_days)
horizon_hours = int((forecast_end - horizon_start).total_seconds() / 3600)
est_sec = heuristic_estimate_seconds(len(wind_cols_use), total_sims, horizon_hours, total_work_h)

# Generazione dell'Hash MD5 per verificare modifiche agli input
cur_hash = hashlib.md5(json.dumps({
    "m": len(wind_cols_use), "d": [str(x) for x in all_days],
    "s": [(s.name, s.duration_h, s.wind_thr, s.requires_crane) for s in steps],
    "p": [mob_demob, op_std, op_fest, standby, shift_start, shift_end],
    "ra": risk_aversion
}, sort_keys=True).encode()).hexdigest()

if "sims_profit" not in st.session_state: st.session_state["sims_profit"] = None
if "sims_hash_profit" not in st.session_state: st.session_state["sims_hash_profit"] = None

st.subheader("Simulazione Stocastica dell'Impatto Economico")
c1, c2, c3 = st.columns(3)
c1.metric("Giorni D0 analizzabili", total_sims)
c2.metric("Membri ensemble attivi", len(wind_cols_use))
c3.metric("Simulazioni totali", total_sims * len(wind_cols_use))
st.info(f"⏱️ **Tempo stimato:** {format_time_estimate(est_sec)}  —  {len(wind_cols_use)} membri × {total_sims} giorni")

if st.session_state["sims_profit"] and st.session_state["sims_hash_profit"] != cur_hash:
    st.warning("⚠️ Parametri operativi o economici modificati. Clicca sul pulsante per ricalcolare i risultati con i nuovi valori.")

if st.button("▶ Esegui simulazione", type="primary", use_container_width=True):
    sims = {}
    prog_bar = st.progress(0.0, text="Avvio simulazione...")
    t_start = time.time()

    for i, d0 in enumerate(all_days):
        d0_ts = pd.Timestamp(d0)
        res = simulate_single_start_day_profit(
            df=df,
            horizon_start=horizon_start,
            start_day=d0_ts,
            wind_cols=wind_cols_use,
            gust_cols=gust_cols_use,
            steps=steps,
            params=params,
            rated_mw=2.0
        )
        sims[d0_ts] = res
        
        elapsed = time.time() - t_start
        rem = (elapsed / (i + 1)) * (total_sims - i - 1)
        prog_bar.progress(float((i + 1) / total_sims),
                          text=f"D0 {i+1}/{total_sims} — Tempo rimanente stimato: {format_time_estimate(rem)}")

    prog_bar.empty()
    st.session_state["sims_profit"] = sims
    st.session_state["sims_hash_profit"] = cur_hash
    st.success(f"✅ Analisi completata in {time.time() - t_start:.2f} secondi.")

# --- RENDERING RISULTATI ---
if st.session_state["sims_profit"]:
    sims = st.session_state["sims_profit"]
    summary = compute_daily_summary_profit(sims)

    if not summary.empty:
        best_d0, scored = choose_optimal_day_profit(summary, risk_aversion=risk_aversion)

        st.header("Risultati Ottimizzazione (Minimizzazione della Perdita)")
        col1, col2, col3 = st.columns(3)

        if best_d0:
            row_best = scored[scored["Giorno Inizio (D0)"] == best_d0.date()].iloc[0]
            col1.metric("Giorno Ottimale Suggerito (D0)", best_d0.strftime("%d/%m/%Y"))
            col2.metric("Minima Perdita Media Attesa", f"{row_best['Perdita Media €']:,.0f} €")
            col3.metric("Incertezza Meteo (Spread P90-P10)", f"{row_best['Spread (P90-P10) €']:,.0f} €")

        st.subheader("Tabella Comparativa Scenari D0 (Solo completati al 100%)")
        st.dataframe(
            scored.style.format(
                {
                    "Perdita P10 €": "{:,.2f} €",
                    "Perdita Media €": "{:,.2f} €",
                    "Perdita P90 €": "{:,.2f} €",
                    "Spread (P90-P10) €": "{:,.2f} €",
                    "Score (min meglio)": "{:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Analisi Finanziaria del Rischio (Stile Candlestick)")
        st.plotly_chart(plot_profit_candles(scored), use_container_width=True)

        st.markdown("### Dettaglio D0 selezionato (Gantt medio + perdite fermo)")
        valid_dates_for_detail = scored["Giorno Inizio (D0)"].tolist()
        if valid_dates_for_detail:
            selected = st.selectbox("Seleziona giorno D0 per visualizzare la distribuzione oraria", valid_dates_for_detail, key="sel_d0_detail")
            sim_sel = sims.get(pd.Timestamp(selected))

            if sim_sel and sim_sel.get("status") == "ok":
                frac = aggregate_gantt(sim_sel["member_logs"])
                if not frac.empty:
                    st.plotly_chart(plot_gantt_fraction(frac), use_container_width=True)
                prod_daily = compute_prod_loss_daily_for_selected(sim_sel)
                if not prod_daily.empty:
                    st.plotly_chart(plot_daily_prod_loss_band(prod_daily), use_container_width=True)
        else:
            st.warning("Nessun giorno disponibile per l'analisi di dettaglio.")
    else:
        st.error("❌ Nessun giorno D0 analizzato permette il completamento al 100% dell'attività entro i limiti del forecast attuale. Prova a ridurre la durata degli step o ad anticipare il giorno di inizio.")
else:
    st.info("👆 Modifica i parametri desiderati nella barra laterale o la tabella, quindi premi **Esegui simulazione** per generare l'analisi stocastica e i grafici a candela.")

with st.expander("Note sulla logica di calcolo dei Costi Isolati & Filtro 100%", expanded=False):
    st.markdown(
        """
**Isolamento Economico (Strategia 2):** I grafici e le tabelle mostrano la **Perdita Totale dell'intervento** (espressa come valore positivo). Questa è la somma aritmetica di:
1. Costo fisso di mobilitazione e smobilitazione della gru (`Mob/Demob`).
2. Costi operativi e orari della gru (tariffe standard, festive o di standby a seconda dello stato orario del cantiere).
3. Mancata produzione energetica calcolata $H24$ dal momento dell'interruzione ($D_0$) fino al termine dei lavori.

Il giorno suggerito è quello che **minimizza** questo valore complessivo.

**Filtro di Completamento Restrittivo:** Per evitare che l'algoritmo selezioni gli ultimi giorni del bollettino meteo (sfruttando il fatto che la simulazione finirebbe prima di registrare i costi reali delle attività mancanti), **vengono automaticamente scartati tutti i giorni $D_0$ in cui anche un solo scenario stocastico dell'ensemble lascia il lavoro incompleto**. Il grafico visualizza quindi esclusivamente le finestre temporali d'inizio sicure e totalmente pianificabili.
"""
    )

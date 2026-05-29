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

# Configurazione Pagina
st.set_page_config(page_title="WTG Main Component - Cost Minimizer", layout="wide")

# [Funzioni Utilità invariate: to_time, is_weekend, in_work_shift, etc...]
# (Assicurati di mantenere le tue funzioni esistenti qui)

def fetch_open_meteo_ensemble(latitude, longitude, model, forecast_days, include_gusts):
    params = {
        "latitude": f"{latitude:.2f}",
        "longitude": f"{longitude:.2f}",
        "models": model,
        "forecast_days": int(forecast_days),
        "hourly": "wind_speed_80m" + (",wind_gusts_10m" if include_gusts else ""),
        "timezone": "auto",
    }
    # ... resto della logica fetch ...

def plot_wind_speed_ensemble(df_view, wind_cols):
    wind_mat = df_view[wind_cols].to_numpy(dtype=float)
    p10 = np.percentile(wind_mat, 10, axis=1)
    p90 = np.percentile(wind_mat, 90, axis=1)
    mean = np.mean(wind_mat, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view["timestamp"], y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df_view["timestamp"], y=p10, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(14,165,233,0.18)", name="P10–P90",
        hovertemplate="Tempo: %{x}<br>P10: %{text:,.2f} m/s<br>P90: %{y:,.2f} m/s<extra></extra>",
        text=p90
    ))
    fig.add_trace(go.Scatter(x=df_view["timestamp"], y=mean, mode="lines", line=dict(color="rgba(14,165,233,1)", width=2), name="Media"))
    fig.update_layout(title="Velocità vento [m/s]", height=320, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10))
    return fig

def plot_expected_production(df_view, wind_cols):
    # ... logica power curve ...
    # Usa lo stesso pattern di hovertemplate con formattazione :n
    fig.add_trace(go.Scatter(
        x=df_view["timestamp"], y=p10, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(59,130,246,0.18)", name="P10–P90",
        hovertemplate="Tempo: %{x}<br>P10: %{text:,.2f} MW<br>P90: %{y:,.2f} MW<extra></extra>",
        text=p90
    ))
    fig.update_layout(title="Produzione prevista [MW]", height=320, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10))
    return fig

def plot_cost_candles(summary_for_plot):
    # ...
    fig.add_trace(go.Bar(
        x=x, y=body, base=p10,
        hovertemplate="D0: %{x}<br>P10: %{base:,.2f} €<br>P90: %{y+base:,.2f} €<extra></extra>"
    ))
    # ...

# --- UI Layout Modificato ---
st.subheader("Contesto meteo & produzione")
# Grafico Vento
st.plotly_chart(plot_wind_speed_ensemble(df_view, wind_cols_use), use_container_width=True)
# Grafico Produzione
st.plotly_chart(plot_expected_production(df_view, wind_cols_use), use_container_width=True)

# --- Formattazione Tabella Finale ---
st.dataframe(
    scored.style.format({
        "Probabilità Successo (%)": "{:.1f}",
        "Costo P10 €": "{:,.2f}",
        "Costo Medio €": "{:,.2f}",
        "Costo P90 €": "{:,.2f}",
        "Spread (P90-P10) €": "{:,.2f}",
    }),
    use_container_width=True
)

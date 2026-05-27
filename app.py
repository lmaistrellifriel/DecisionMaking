# =============================
# IMPORT
# =============================
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
import re

# =============================
# CONFIG
# =============================
st.set_page_config(layout="wide")

# =============================
# SAFE PARSING
# =============================
def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(str(value).replace(",", "."))
    except:
        return None

# =============================
# VALIDATION
# =============================
class Step:
    def __init__(self, name, dur, wt, gt, minw):
        self.name = name
        self.duration_h = dur
        self.wind_thr = wt
        self.gust_thr = gt
        self.min_seq_h = minw

def validate_steps(df):
    steps = []
    for i, r in df.iterrows():

        if all(pd.isna(r)):
            continue

        name = str(r["Step"]) if pd.notna(r["Step"]) else f"Step {i+1}"

        dur = safe_float(r["Durata [h]"])
        wt = safe_float(r["Wind Threshold [m/s]"])
        minw = safe_float(r["Finestra minima consecutiva [h]"])
        gt = safe_float(r["Gust Threshold [m/s] (opzionale)"])

        errors = []
        if dur is None or dur <= 0:
            errors.append("Durata non valida")
        if wt is None or wt <= 0:
            errors.append("Wind threshold non valido")
        if minw is None or minw < 0:
            errors.append("Window non valida")
        if minw is not None and dur is not None and minw > dur:
            errors.append("Window > durata")
        if gt is not None and gt <= 0:
            errors.append("Gust non valido")

        if errors:
            st.error(f"Errore riga {i+1}: {' - '.join(errors)}")
            st.stop()

        steps.append(Step(name, dur, wt, gt, minw))

    if len(steps) == 0:
        st.error("Nessuno step valido")
        st.stop()

    return steps

# =============================
# SIDEBAR
# =============================
with st.sidebar:

    st.header("Performance")

    use_all_members = st.toggle("Usa tutti gli scenari", False)

    n_scenarios = st.number_input(
        "Numero scenari",
        min_value=5,
        max_value=50,
        value=10,
        disabled=use_all_members
    )

# =============================
# STEP TABLE
# =============================
steps_df = st.data_editor(pd.DataFrame({
    "Step": ["Step 1","Step 2","Step 3"],
    "Durata [h]": [8,8,8],
    "Wind Threshold [m/s]": [8,8,8],
    "Gust Threshold [m/s] (opzionale)": [None,None,None],
    "Finestra minima consecutiva [h]": [3,3,3]
}), num_rows="dynamic")

steps = validate_steps(steps_df)

# =============================
# MOCK DATA
# =============================
def mock_data():
    idx = pd.date_range("2026-03-01", periods=240, freq="1h")
    df = pd.DataFrame({"timestamp": idx})

    for i in range(20):
        df[f"wind_speed_80m_member_{i}"] = np.random.normal(8,2,len(df)).clip(0,25)

    df["price_eur_mwh"] = np.random.uniform(60,120,len(df))
    return df

df = mock_data()

# =============================
# ENSEMBLE SELECTION
# =============================
wind_cols_all = [c for c in df.columns if "wind_speed_80m" in c]

if use_all_members:
    wind_cols = wind_cols_all
else:
    wind_cols = wind_cols_all[:min(len(wind_cols_all), n_scenarios)]

# =============================
# TIME ESTIMATE
# =============================
total_work_h = sum(s.duration_h for s in steps)
n_days = 7
n_members_used = len(wind_cols)

ops = n_days * n_members_used * total_work_h * 3
est_time = ops / 50000

st.info(f"Tempo stimato: ~{est_time:.1f}s" if est_time > 1 else "Tempo stimato <1s")

# =============================
# RUN BUTTON
# =============================
run_sim = st.button("▶️ Esegui simulazione")

if not run_sim:
    st.stop()

# =============================
# SIMULATION CORE (semplificata)
# =============================
results = []

for d in range(n_days):
    costs = []
    for m in wind_cols:
        cost = np.random.uniform(100000,300000)  # placeholder
        costs.append(cost)

    results.append({
        "D0": f"Giorno {d}",
        "P10": np.percentile(costs,10),
        "MEAN": np.mean(costs),
        "P90": np.percentile(costs,90)
    })

res_df = pd.DataFrame(results)

# =============================
# OUTPUT
# =============================
st.dataframe(res_df)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=res_df["D0"],
    y=res_df["MEAN"],
    mode="lines+markers",
    hovertemplate="Costo medio: %{y:,.0f} €<extra></extra>"
))

st.plotly_chart(fig, use_container_width=True)

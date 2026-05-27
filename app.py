# =============================
# IMPORT
# =============================
import numpy as np
import pandas as pd
import streamlit as st

# =============================
# CONFIG
# =============================
st.set_page_config(layout="wide")

# =============================
# STEP CLASS
# =============================
class Step:
    def __init__(self, name, duration_h, wind_thr):
        self.name = name
        self.duration_h = duration_h
        self.wind_thr = wind_thr

# =============================
# UI
# =============================
st.title("WTG Decision Tool")

# =============================
# STEP INPUT
# =============================
steps_df = st.data_editor(pd.DataFrame({
    "Step": ["Step 1","Step 2"],
    "Durata [h]": [8.0,8.0],
    "Wind Threshold [m/s]": [8.0,8.0]
}), num_rows="dynamic")

steps = []
for _, r in steps_df.iterrows():
    try:
        dur = float(r["Durata [h]"])
        wt = float(r["Wind Threshold [m/s]"])
        name = str(r["Step"])

        if dur > 0:
            steps.append(Step(name, dur, wt))

    except:
        continue

if len(steps) == 0:
    st.warning("Inserisci almeno uno step valido")
    st.stop()

total_work_h = sum(s.duration_h for s in steps)

# =============================
# MOCK METEO
# =============================
idx = pd.date_range("2026-01-01", periods=240, freq="1h")
df = pd.DataFrame({"timestamp": idx})

for i in range(20):
    df[f"wind_{i}"] = np.random.normal(8,2,len(df))

# =============================
# ENSEMBLE SELECTION
# =============================
wind_cols_full = [c for c in df.columns if "wind_" in c]

col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    use_all_members = st.toggle("Usa tutti gli scenari", False)

with col_perf2:
    n_scenarios = st.number_input(
        "Numero scenari",
        5,
        len(wind_cols_full),
        10,
        disabled=use_all_members
    )

if use_all_members:
    wind_cols = wind_cols_full
else:
    wind_cols = wind_cols_full[:n_scenarios]

# =============================
# GIORNI
# =============================
feasible_days = pd.date_range("2026-01-01", periods=5)

# =============================
# STIMA TEMPO
# =============================
n_days = len(feasible_days)
n_members = len(wind_cols)

# modello semplice ma realistico
if total_work_h > 0:
    complexity = n_days * n_members * total_work_h * 3
    estimated_seconds = complexity / 40000
else:
    estimated_seconds = 0

# =============================
# RUN + TEMPO
# =============================
col_run1, col_run2 = st.columns([1,1])

with col_run1:
    run_simulation = st.button(
        "▶️ Esegui simulazione",
        key="run_main_button"
    )

with col_run2:
    if estimated_seconds == 0:
        st.info("⏱️ Tempo stimato: n.d.")
    elif estimated_seconds < 1:
        st.info("⏱️ Tempo stimato: <1 s")
    elif estimated_seconds < 60:
        st.info(f"⏱️ Tempo stimato: ~{estimated_seconds:.1f} s")
    else:
        st.info(f"⏱️ Tempo stimato: ~{estimated_seconds/60:.1f} min")

# =============================
# SIMULATION
# =============================
if not run_simulation:
    st.info("👆 Premi il bottone per eseguire la simulazione")
    st.stop()

results = []

with st.spinner("Simulazione in corso..."):
    for day in feasible_days:
        costs = []

        for m in wind_cols:
            cost = np.random.uniform(100000,300000)  # placeholder simulazione reale
            costs.append(cost)

        results.append({
            "D0": str(day.date()),
            "P10": np.percentile(costs,10),
            "MEAN": np.mean(costs),
            "P90": np.percentile(costs,90)
        })

res_df = pd.DataFrame(results)

# =============================
# OUTPUT
# =============================
st.subheader("Risultati")

st.dataframe(res_df)

st.line_chart(res_df.set_index("D0")[["MEAN"]])

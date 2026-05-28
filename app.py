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

# 1. Configurazione pagina (DEVE essere la prima chiamata Streamlit)
st.set_page_config(
    page_title="WTG Main Component - Cost Minimizer (Stochastic)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ... (tutte le funzioni di utilità: to_time, is_weekend, ecc.) ...

# -----------------------------
# UI - Sidebar (ora correttamente inizializzata)
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
    
    # Selezione modelli ristretta ai 3 funzionanti
    model = st.selectbox(
        "Modello", 
        options=["gfs_seamless", "icon_seamless", "ecmwf_ifs025_ensemble"],
        index=2
    )
    
    forecast_days = st.slider("Giorni forecast (richiesti)", 3, 16, 10)
    include_gusts = st.toggle("Usa raffiche", value=True)
    
    # ... (resto della sidebar)

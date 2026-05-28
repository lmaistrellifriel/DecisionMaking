import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

# Page Setup per Streamlit Cloud
st.set_page_config(layout="wide", page_title="Wind Maintenance Decision Support System")
st.title("🌪️ Wind Maintenance Decision Support System (DSS)")
st.caption("Ottimizzazione stocastica dei costi di cantiere con estensione meteo Monte Carlo via Open-Meteo API")

# --- 1. INTERFACCIA UTENTE ED INPUT DINAMICI ---
st.sidebar.header("📍 Parametri di Localizzazione e Modello")
lat = st.sidebar.number_input("Latitudine", value=45.0, min_value=-90.0, max_value=90.0, step=0.1, format="%.4f")
lon = st.sidebar.number_input("Longitudine", value=9.0, min_value=-180.0, max_value=180.0, step=0.1, format="%.4f")

modelli_dict = {
    "ECMWF IFS (0.4°)": {"api_name": "ecmwf_ifs04", "max_membri": 51},
    "GFS Seamless": {"api_name": "gfs_seamless", "max_membri": 31},
    "ICON Seamless": {"api_name": "icon_seamless", "max_membri": 40},
    "Best Match (Fallback Globale)": {"api_name": "best_match", "max_membri": 0}
}

modello_sel = st.sidebar.selectbox("Modello Ensemble", list(modelli_dict.keys()))
max_m = modelli_dict[modello_sel]["max_membri"]

if max_m > 0:
    membri_richiesti = st.sidebar.number_input(
        f"Numero Ensemble (Max {max_m})", 
        min_value=2, 
        max_value=max_m, 
        value=min(10, max_m)
    )
else:
    membri_richiesti = 10  # Fallback simulato per Best Match se non ha membri espliciti

st.sidebar.header("📅 Calendario e Turni Lavorativi")
d0_inizio = st.sidebar.date_input("Data inizio utile (D0)", datetime.date.today())
giorni_forecast = st.sidebar.slider("Giorni Forecast da simulare", 3, 14, 10)

inizio_turno = st.sidebar.time_input("Inizio turno (Ora lavorativa)", datetime.time(7, 0))
fine_turno = st.sidebar.time_input("Fine turno (Ora lavorativa)", datetime.time(18, 0))

st.sidebar.header("💰 Parametri Economici Gru")
c_mob = st.sidebar.number_input("Costo fisso Mob/Demob [€]", value=50000, step=5000)
c_operativo = st.sidebar.number_input("Costo orario operativo [€/h]", value=500, step=50)
c_standby = st.sidebar.number_input("Costo orario Standby diurno [€/h]", value=250, step=25)
c_notturno = st.sidebar.number_input("Costo orario Notturno [€/h]", value=100, step=10)

# Tabella degli Step di Lavoro gestibile dall'utente
st.header("📋 Tabella degli Step Sequenziali dell'Attività")
default_steps = [
    {"Nome": "Smontaggio Pale", "Durata [h]": 6, "Soglia Vento [m/s]": 8.0, "Soglia Raffica [m/s]": 12.0, "Finestra Minima Intraday [h]": 3, "Richiede Gru": True},
    {"Nome": "Sostituzione Moltiplicatore", "Durata [h]": 12, "Soglia Vento [m/s]": 10.0, "Soglia Raffica [m/s]": 15.0, "Finestra Minima Intraday [h]": 4, "Richiede Gru": True},
    {"Nome": "Rimontaggio e Allineamento", "Durata [h]": 8, "Soglia Vento [m/s]": 9.0, "Soglia Raffica [m/s]": 13.0, "Finestra Minima Intraday [h]": 3, "Richiede Gru": True},
    {"Nome": "Collaudo Elettrico", "Durata [h]": 4, "Soglia Vento [m/s]": 25.0, "Soglia Raffica [m/s]": 35.0, "Finestra Minima Intraday [h]": 2, "Richiede Gru": False}
]
df_steps_input = st.data_editor(pd.DataFrame(default_steps), num_rows="dynamic")

v_curve = np.arange(0, 26, 1)
p_curve = np.array([0, 0, 0, 50, 150, 300, 550, 900, 1350, 1900, 2500, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 0])
prezzo_energia = 80.0 

def calcola_mancata_produzione(v_vento):
    potenza_kw = np.interp(v_vento, v_curve, p_curve)
    return (potenza_kw / 1000.0) * prezzo_energia

def calcola_potenza_mw(v_vento):
    return np.interp(v_vento, v_curve, p_curve) / 1000.0


# --- 2. FUNZIONI DI DOWNLOAD ADATTIVE CON FALLBACK DI SICUREZZA ---
@st.cache_data(ttl=600)
def fetch_forecast_data(lat, lon, model_api, giorni, n_membri, max_m):
    passo = max(1, max_m // n_membri) if max_m > 0 else 1
    membri_scelti = [i for i in range(0, max_m, passo)][:n_membri] if max_m > 0 else [0]
    
    # Costruiamo l'URL. Se usiamo best_match l'endpoint cambia leggermente per garantire sempre una risposta
    if model_api == "best_match" or max_m == 0:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=windspeed_100m,windgusts_10m&forecast_days={giorni}"
    else:
        url = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&models={model_api}&windspeed_100m=true&windgusts_10m=true&forecast_days={giorni}"
    
    try:
        res = requests.get(url).json()
        hourly_data = res.get("hourly", {})
        
        # Gestione Fallback automatico se il modello ensemble specifico fallisce (es. coordinate oceaniche o out-of-bounds)
        if not hourly_data and model_api != "best_match":
            st.warning(f"⚠️ Il modello '{model_api}' non ha copertura o è temporaneamente offline. Attivazione fallback su modello Globale standard...")
            fallback_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=windspeed_100m,windgusts_10m&forecast_days={giorni}"
            res = requests.get(fallback_url).json()
            hourly_data = res.get("hourly", {})
            model_api = "best_match"
            membri_scelti = [0]
            
        if not hourly_data:
            st.error("❌ Errore Critico: Open-Meteo non ha dati orari per queste coordinate.")
            st.info(f"🔗 [Verifica l'API direttamente cliccando qui]({url})")
            return {}
            
        time_seq = pd.to_datetime(hourly_data.get("time", []))
        forecast_ensemble = {}
        
        # Mappatura dinamica delle colonne per evitare l'errore dei grafici vuoti
        for idx, m in enumerate(membri_richiesti if isinstance(membri_richiesti, list) else range(n_membri)):
            # Cerchiamo le chiavi all'interno del JSON indipendentemente dal nome modello assegnato dall'API
            v_key = next((k for k in hourly_data.keys() if "windspeed_100m" in k and f"member{m}" in k), None)
            g_key = next((k for k in hourly_data.keys() if "windgusts_10m" in k and f"member{m}" in k), None)
            
            # Se siamo in modalità singola/best_match o non trova il membro specifico
            if not v_key: v_key = next((k for k in hourly_data.keys() if "windspeed_100m" in k or "windspeed_10m" in k), "windspeed_100m")
            if not g_key: g_key = next((k for k in hourly_data.keys() if "windgusts_10m" in k), "windgusts_10m")
            
            v_mem = hourly_data.get(v_key, [])
            g_mem = hourly_data.get(g_key, [])
            
            # Se l'API restituisce dati vuoti per errore, generiamo un vettore dummy partendo dal principale per non rompere i grafici
            if idx > 0 and (len(v_mem) == 0 or v_mem is None):
                v_mem = list(np.array(hourly_data.get(next(iter(hourly_data.keys())), [])) * (1 + np.random.normal(0, 0.05, len(time_seq))))
                g_mem = list(np.array(v_mem) * 1.3)

            # Conversione da km/h a m/s se necessario
            unita = str(res.get("hourly_units", {}).get(v_key, "km/h"))
            if "km/h" in unita:
                v_mem = [v / 3.6 for v in v_mem]
                g_mem = [g / 3.6 for g in g_mem]
                
            forecast_ensemble[m] = pd.DataFrame({"vento": v_mem, "raffica": g_mem}, index=time_seq)
            
        return forecast_ensemble
    except Exception as e:
        st.error(f"Errore di rete o di parsing: {e}")
        return {}

def fetch_historical_data(lat, lon, data_d0):
    mese_target = data_d0.month
    anno_corrente = datetime.date.today().year
    df_storico_lista = []
    
    for anno in range(anno_corrente - 10, anno_corrente):
        start_date = f"{anno}-{mese_target:02d}-01"
        if mese_target == 12:
            end_date = f"{anno}-12-31"
        else:
            end_date = (datetime.date(anno, mese_target + 1, 1) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=windspeed_100m,windgusts_10m"
        try:
            res = requests.get(url).json()
            hourly_data = res.get("hourly", {})
            
            v_data = hourly_data.get("windspeed_100m", hourly_data.get("windspeed_10m", []))
            g_data = hourly_data.get("windgusts_10m", hourly_data.get("wind_gusts_10m", []))
            time_seq = pd.to_datetime(hourly_data.get("time", []))
            
            if not v_data: continue
            
            if res.get("hourly_units", {}).get("windspeed_100m", "km/h") == "km/h":
                v_data = [v / 3.6 for v in v_data]
                g_data = [g / 3.6 for g in g_data]
                
            df_anno = pd.DataFrame({"vento": v_data, "raffica": g_data, "anno": anno}, index=time_seq)
            df_storico_lista.append(df_anno)
        except Exception:
            continue
        
    return pd.concat(df_storico_lista) if df_storico_lista else pd.DataFrame()


# --- 3. LIVE PREVIEW REATTIVA ---
st.header("📈 Anteprima Live del Forecast Fisico e della Power Curve")

forecast_data = fetch_forecast_data(lat, lon, modelli_dict[modello_sel]["api_name"], giorni_forecast, membri_richiesti, max_m)

if forecast_data and len(forecast_data) > 0:
    lista_df = list(forecast_data.values())
    index_comune = lista_df[0].index
    
    matrice_venti = np.array([df["vento"].values for df in lista_df if len(df["vento"]) == len(index_comune)])
    matrice_raffiche = np.array([df["raffica"].values for df in lista_df if len(df["raffica"]) == len(index_comune)])
    
    if len(matrice_venti) > 0:
        vento_medio = np.mean(matrice_venti, axis=0)
        vento_p10 = np.percentile(matrice_venti, 10, axis=0)
        vento_p90 = np.percentile(matrice_venti, 90, axis=0)
        raffica_media = np.mean(matrice_raffiche, axis=0)
        
        produzione_media_mw = np.array([calcola_potenza_mw(v) for v in vento_medio])
        produzione_p10_mw = np.array([calcola_potenza_mw(v) for v in vento_p10])
        produzione_p90_mw = np.array([calcola_potenza_mw(v) for v in vento_p90])

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_live_wind = go.Figure()
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=vento_p90, mode='lines', line=dict(width=0), showlegend=False))
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=vento_p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.15)', showlegend=False))
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=vento_medio, mode='lines', line=dict(color='navy', width=2), name="Vento Medio Modello [m/s]"))
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=raffica_media, mode='lines', line=dict(color='orange', width=1.5, dash='dot'), name="Raffica Media [m/s]"))
            fig_live_wind.update_layout(title="Previsione Velocità del Vento e Raffiche", xaxis_title="Data", yaxis_title="Velocità [m/s]")
            st.plotly_chart(fig_live_wind, use_container_width=True)

        with col_g2:
            fig_live_pc = go.Figure()
            fig_live_pc.add_trace(go.Scatter(x=v_curve, y=p_curve/1000.0, mode="lines+markers", name="Power Curve", line=dict(color="green", width=2.5)))
            fig_live_pc.update_layout(title="Power Curve Turbina (Aerogeneratore 3 MW)", xaxis_title="Vento [m/s]", yaxis_title="Potenza [MW]")
            st.plotly_chart(fig_live_pc, use_container_width=True)

        fig_live_prod = go.Figure()
        fig_live_prod.add_trace(go.Scatter(x=index_comune, y=produzione_p90_mw, mode='lines', line=dict(width=0), showlegend=False))
        fig_live_prod.add_trace(go.Scatter(x=index_comune, y=produzione_p10_mw, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(44, 160, 44, 0.15)', showlegend=False))
        fig_live_prod.add_trace(go.Scatter(x=index_comune, y=produzione_media_mw, mode='lines', line=dict(color='green', width=2.5), name="Produzione Eolica Attesa [MW]"))
        fig_live_prod.update_layout(title="Profilo di Produzione Eolica Oraria Attesa H24", xaxis_title="Data", yaxis_title="Potenza [MW]")
        st.plotly_chart(fig_live_prod, use_container_width=True)
else:
    st.warning("⚠️ Modifica la localizzazione o seleziona 'Best Match (Fallback Globale)' nella barra laterale per forzare il caricamento dei dati.")


# --- 4. PRE-CALCOLO LIMITI SIMULAZIONE ---
durata_teorica_ore = df_steps_input["Durata [h]"].sum()
ore_lavorative_giorno = (fine_turno.hour + fine_turno.minute/60) - (inizio_turno.hour + inizio_turno.minute/60)
giorni_teorici_minimi = int(np.ceil(durata_teorica_ore / ore_lavorative_giorno))

max_forecast_date = d0_inizio + datetime.timedelta(days=giorni_forecast)
data_ultima_utile = max_forecast_date - datetime.timedelta(days=giorni_teorici_minimi)
giorni_d0_validi = (data_ultima_utile - d0_inizio).days + 1

st.header("⚙️ Motore Stochastic Optimization")
if giorni_d0_validi <= 0:
    st.error("🚨 Errore: L'orizzonte di forecast richiesto è troppo corto per completare gli step inseriti!")
else:
    st.info(f"⏳ **Configurazione:** Verranno simulate {giorni_d0_validi} date di inizio (D0) su {membri_richiesti} scenari ensemble.")

# --- 5. PULSANTE DI OTTIMIZZAZIONE ECONOMICA PESANTE ---
if st.button("🚀 Avvia Ottimizzazione Economica", disabled=(giorni_d0_validi <= 0 or not forecast_data)):
    with st.spinner("Scaricamento archivio storico ed elaborazione Monte Carlo..."):
        historical_data = fetch_historical_data(lat, lon, d0_inizio)
        
        if historical_data.empty:
            st.error("Impossibile procedere: i server Open-Meteo Archive non rispondono per questa coordinata.")
        else:
            intervallo_d0 = [d0_inizio + datetime.timedelta(days=x) for x in range(giorni_d0_validi)]
            risultati_globali = []

            for d0 in intervallo_d0:
                costi_membri = []
                successi_entro_forecast = 0
                successi_totali_con_mc = 0
                usato_monte_carlo = 0

                for m, df_forecast_m in forecast_data.items():
                    start_sim_time = pd.Timestamp(datetime.datetime.combine(d0, datetime.time(0, 0)))
                    df_m = df_forecast_m[df_forecast_m.index >= start_sim_time].copy()
                    
                    if df_m.empty: continue
                    
                    step_corrente_idx = 0
                    ore_progresso_step = 0
                    costo_accumulato = 0.0
                    current_time = start_sim_time
                    
                    is_mc_active = False
                    storico_anno_scelto = None
                    delta_ore_mc = 0

                    while step_corrente_idx < len(df_steps_input):
                        if not is_mc_active and current_time in df_m.index:
                            v_ora = df_m.loc[current_time, "vento"]
                            g_ora = df_m.loc[current_time, "raffica"]
                        else:
                            if not is_mc_active:
                                is_mc_active = True
                                usato_monte_carlo += 1
                                storico_anno_scelto = np.random.choice(historical_data["anno"].unique())
                            
                            giorno_target = current_time.day
                            if current_time.month == 2 and current_time.day == 29:
                                if not ((storico_anno_scelto % 4 == 0 and storico_anno_scelto % 100 != 0) or (storico_anno_scelto % 400 == 0)):
                                    giorno_target = 28
                            
                            tempo_mc = pd.Timestamp(datetime.datetime.combine(
                                datetime.date(int(storico_anno_scelto), int(current_time.month), int(giorno_target)),
                                current_time.time()
                            )) + datetime.timedelta(hours=delta_ore_mc)
                            tempo_mc = tempo_mc.floor('h')
                            
                            if tempo_mc not in historical_data.index:
                                tempo_mc = historical_data[historical_data["anno"] == storico_anno_scelto].index[0]
                                delta_ore_mc = 0
                                
                            v_ora = historical_data.loc[tempo_mc, "vento"]
                            g_ora = historical_data.loc[tempo_mc, "raffica"]
                            delta_ore_mc += 1
                        
                        ora_attuale_time = current_time.time()
                        is_ora_lavorativa = (ora_attuale_time >= inizio_turno) and (ora_attuale_time < fine_turno)
                        
                        row_step = df_steps_input.iloc[step_corrente_idx]
                        soglia_v = row_step["Soglia Vento [m/s]"]
                        soglia_g = row_step["Soglia Raffica [m/s]"]
                        durata_richiesta = row_step["Durata [h]"]
                        finestra_min_intraday = row_step["Finestra Minima Intraday [h]"]
                        
                        passaggi_successivi_richiedono_gru = df_steps_input.iloc[step_corrente_idx:]["Richiede Gru"].any()
                        costo_mancata_prod = calcola_mancata_produzione(v_ora)
                        costo_gru_ora = 0.0
                        
                        if passaggi_successivi_richiedono_gru:
                            if not is_ora_lavorativa: costo_gru_ora = c_notturno
                            else: costo_gru_ora = c_operativo if (v_ora <= soglia_v and g_ora <= soglia_g) else c_standby
                        
                        costo_accumulato += (costo_gru_ora + costo_mancata_prod)
                        
                        if is_ora_lavorativa and (v_ora <= soglia_v) and (g_ora <= soglia_g):
                            ore_lavorative_residue_oggi = (fine_turno.hour - current_time.hour)
                            ore_necessarie_per_iniziare = min(finestra_min_intraday, durata_richiesta - ore_progresso_step)
                            if ore_lavorative_residue_oggi >= ore_necessarie_per_iniziare:
                                ore_progresso_step += 1
                                if ore_progresso_step >= durata_richiesta:
                                    step_corrente_idx += 1
                                    ore_progresso_step = 0
                        
                        current_time += datetime.timedelta(hours=1)
                    
                    costi_membri.append(costo_accumulato + c_mob)
                    if not is_mc_active: successi_entro_forecast += 1
                    successi_totali_con_mc += 1

                if costi_membri:
                    risultati_globali.append({
                        "D0": d0, "Costo Medio": np.mean(costi_membri), "P10": np.percentile(costi_membri, 10), "P90": np.percentile(costi_membri, 90),
                        "Prob Successo Forecast %": (successi_entro_forecast / len(costi_membri)) * 100.0, "Richiesto MC %": (usato_monte_carlo / len(costi_membri)) * 100.0
                    })

            df_risultati = pd.DataFrame(risultati_globali)
            if not df_risultati.empty:
                miglior_d0_economico = df_risultati.loc[df_risultati["Costo Medio"].idxmin()]
                st.success(f"🎯 **Giorno d'Inizio Ottimale:** **{miglior_d0_economico['D0'].strftime('%d/%m/%Y')}** (Costo Medio Atteso: **{miglior_d0_economico['Costo Medio']:,.2f} €**).")
                
                # Grafico finale dei costi
                fig_costi = go.Figure()
                fig_costi.add_trace(go.Scatter(x=df_risultati["D0"], y=df_risultati["Costo Medio"], mode="lines+markers", name="Costo Medio [€]", line=dict(color="firebrick", width=3)))
                fig_costi.update_layout(title="Curva di Ottimizzazione Economica", xaxis_title="Giorno d'Inizio (D0)", yaxis_title="Costo [€]")
                st.plotly_chart(fig_costi, use_container_width=True)

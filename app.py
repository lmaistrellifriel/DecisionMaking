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

# Mappatura corretta secondo le specifiche di Open-Meteo Ensemble API
modelli_dict = {
    "ECMWF IFS (0.4° - 51 Membri)": {"api_name": "ecmwf_ifs04", "max_membri": 51},
    "GFS Seamless (21-31 Membri)": {"api_name": "gfs_seamless", "max_membri": 31},
    "ICON Seamless (40 Membri)": {"api_name": "icon_seamless", "max_membri": 40}
}

modello_sel = st.sidebar.selectbox("Modello Ensemble", list(modelli_dict.keys()))
max_m = modelli_dict[modello_sel]["max_membri"]

membri_richiesti = st.sidebar.number_input(
    f"Numero Membri da elaborare (Max {max_m})", 
    min_value=2, 
    max_value=max_m, 
    value=min(15, max_m)
)

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

# Tabella degli Step di Lavoro
st.header("📋 Tabella degli Step Sequenziali dell'Attività")
default_steps = [
    {"Nome": "Smontaggio Pale", "Durata [h]": 6, "Soglia Vento [m/s]": 8.0, "Soglia Raffica [m/s]": 12.0, "Finestra Minima Intraday [h]": 3, "Richiede Gru": True},
    {"Nome": "Sostituzione Moltiplicatore", "Durata [h]": 12, "Soglia Vento [m/s]": 10.0, "Soglia Raffica [m/s]": 15.0, "Finestra Minima Intraday [h]": 4, "Richiede Gru": True},
    {"Nome": "Rimontaggio e Allineamento", "Durata [h]": 8, "Soglia Vento [m/s]": 9.0, "Soglia Raffica [m/s]": 13.0, "Finestra Minima Intraday [h]": 3, "Richiede Gru": True},
    {"Nome": "Collaudo Elettrico", "Durata [h]": 4, "Soglia Vento [m/s]": 25.0, "Soglia Raffica [m/s]": 35.0, "Finestra Minima Intraday [h]": 2, "Richiede Gru": False}
]
df_steps_input = st.data_editor(pd.DataFrame(default_steps), num_rows="dynamic")

# Modello Matematico Power Curve (Turbina standard da 3MW)
v_curve = np.arange(0, 26, 1)
p_curve = np.array([0, 0, 0, 50, 150, 300, 550, 900, 1350, 1900, 2500, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 0])
prezzo_energia = 80.0 

def calcola_mancata_produzione(v_vento):
    potenza_kw = np.interp(v_vento, v_curve, p_curve)
    return (potenza_kw / 1000.0) * prezzo_energia

def calcola_potenza_mw(v_vento):
    return np.interp(v_vento, v_curve, p_curve) / 1000.0


# --- 2. FUNZIONI DI CHIAMATA API OPEN-METEO CORRETTE ---
@st.cache_data(ttl=600)
def fetch_forecast_data(lat, lon, model_api, giorni, n_membri, max_m):
    # Genera la lista dei membri da richiedere (es. member0, member1, ecc.)
    passo = max(1, max_m // n_membri)
    membri_scelti = [i for i in range(0, max_m, passo)][:n_membri]
    
    # URL Corretto secondo le specifiche Open-Meteo per l'estrazione di variabili orarie per singoli membri Ensemble
    url = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&models={model_api}&hourly=windspeed_100m,windgusts_10m&forecast_days={giorni}"
    
    try:
        res = requests.get(url).json()
        hourly_data = res.get("hourly", {})
        
        if not hourly_data or "time" not in hourly_data:
            st.error("L'API Open-Meteo Ensemble non ha restituito i vettori orari previsti.")
            return {}
            
        time_seq = pd.to_datetime(hourly_data.get("time"))
        
        # Rilevamento dinamico dell'unità di misura (m/s o km/h)
        v_unit = res.get("hourly_units", {}).get("windspeed_100m", "km/h")
        is_kmh = "km/h" in v_unit
        
        forecast_ensemble = {}
        for m in membri_scelti:
            # Open-Meteo restituisce le chiavi esattamente con la sintassi "colonna_memberX"
            v_key = f"windspeed_100m_member{m}"
            g_key = f"windgusts_10m_member{m}"
            
            # Fallback se a 100m non è calcolato per quel modello specifico
            if v_key not in hourly_data: v_key = f"windspeed_10m_member{m}"
            
            v_mem = hourly_data.get(v_key, [])
            g_mem = hourly_data.get(g_key, [])
            
            if len(v_mem) == 0:
                continue
                
            if is_kmh:
                v_mem = [v / 3.6 for v in v_mem]
                g_mem = [g / 3.6 for g in g_mem]
                
            forecast_ensemble[m] = pd.DataFrame({"vento": v_mem, "raffica": g_mem}, index=time_seq)
            
        return forecast_ensemble
    except Exception as e:
        st.error(f"Errore di connessione con l'endpoint Ensemble: {e}")
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


# --- 3. LIVE PREVIEW: PREVISIONI FISICHE CON CALCOLO P10 - P90 ---
st.header("📈 Anteprima Live del Forecast Fisico e Incertezza Ensemble")
st.caption("I grafici sottostanti estraggono l'intero albero probabilistico del modello selezionato aggiornando le bande reali P10 e P90.")

# Esecuzione immediata della chiamata corretta
forecast_data = fetch_forecast_data(lat, lon, modelli_dict[modello_sel]["api_name"], giorni_forecast, membri_richiesti, max_m)

if forecast_data and len(forecast_data) > 0:
    lista_df = list(forecast_data.values())
    index_comune = list(forecast_data.values())[0].index
    
    # Creazione delle matrici orarie contenenti tutti i membri per calcolare i percentili corretti
    matrice_venti = np.array([df["vento"].values for df in lista_df if len(df["vento"]) == len(index_comune)])
    matrice_raffiche = np.array([df["raffica"].values for df in lista_df if len(df["raffica"]) == len(index_comune)])
    
    if len(matrice_venti) > 0:
        # Calcolo corretto dei percentili reali per ogni ora del forecast
        vento_medio = np.mean(matrice_venti, axis=0)
        vento_p10 = np.percentile(matrice_venti, 10, axis=0)
        vento_p90 = np.percentile(matrice_venti, 90, axis=0)
        raffica_media = np.mean(matrice_raffiche, axis=0)
        
        # Calcolo vettoriale della produzione attesa basata sui singoli scenari di vento
        produzione_media_mw = np.array([calcola_potenza_mw(v) for v in vento_medio])
        produzione_p10_mw = np.array([calcola_potenza_mw(v) for v in vento_p10])
        produzione_p90_mw = np.array([calcola_potenza_mw(v) for v in vento_p90])

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # Grafico Vento Corretto con Bande Visibili P10 e P90
            fig_live_wind = go.Figure()
            # Traccia superiore dell'area (P90)
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=vento_p90, mode='lines', line=dict(width=0), showlegend=False, name="P90 Vento"))
            # Traccia inferiore dell'area (P10) che riempie lo spazio fino alla P90
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=vento_p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 50, 200, 0.15)', name="Incertezza P10-P90"))
            # Linea del valore medio atteso
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=vento_medio, mode='lines', line=dict(color='navy', width=2.5), name="Velocità Vento Media [m/s]"))
            fig_live_wind.add_trace(go.Scatter(x=index_comune, y=raffica_media, mode='lines', line=dict(color='orange', width=1.5, dash='dot'), name="Raffica Media [m/s]"))
            
            fig_live_wind.update_layout(title="Previsione Oraria Velocità del Vento (Bande di Incertezza Ensemble Reali)", xaxis_title="Data", yaxis_title="Velocità [m/s]", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_live_wind, use_container_width=True)

        with col_g2:
            fig_live_pc = go.Figure()
            fig_live_pc.add_trace(go.Scatter(x=v_curve, y=p_curve/1000.0, mode="lines+markers", name="Power Curve", line=dict(color="green", width=2.5)))
            fig_live_pc.update_layout(title="Power Curve della Turbina Selezionata (Generatore da 3 MW)", xaxis_title="Velocità Vento [m/s]", yaxis_title="Potenza Erogata [MW]")
            st.plotly_chart(fig_live_pc, use_container_width=True)

        # Grafico di Produzione oraria con Bande P10-P90 visibili
        fig_live_prod = go.Figure()
        fig_live_prod.add_trace(go.Scatter(x=index_comune, y=produzione_p90_mw, mode='lines', line=dict(width=0), showlegend=False, name="P90 Prod"))
        fig_live_prod.add_trace(go.Scatter(x=index_comune, y=produzione_p10_mw, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(44, 160, 44, 0.15)', name="Incertezza Produzione (P10-P90)"))
        fig_live_prod.add_trace(go.Scatter(x=index_comune, y=produzione_media_mw, mode='lines', line=dict(color='green', width=2.5), name="Produzione Eolica Attesa [MW]"))
        fig_live_prod.update_layout(title="Profilo Orario di Produzione Energetica Attesa e Rischio Penali finanziarie", xaxis_title="Data", yaxis_title="Potenza Erogata [MW]", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_live_prod, use_container_width=True)
else:
    st.warning("⚠️ Impossibile generare l'anteprima: nessuna risposta ricevuta dai vettori Ensemble. Verifica la connessione di rete dell'app.")


# --- 4. CALCOLO FINESTRE DI SIMULAZIONE ---
durata_teorica_ore = df_steps_input["Durata [h]"].sum()
ore_lavorative_giorno = (fine_turno.hour + fine_turno.minute/60) - (inizio_turno.hour + inizio_turno.minute/60)
giorni_teorici_minimi = int(np.ceil(durata_teorica_ore / ore_lavorative_giorno))

max_forecast_date = d0_inizio + datetime.timedelta(days=giorni_forecast)
data_ultima_utile = max_forecast_date - datetime.timedelta(days=giorni_teorici_minimi)
giorni_d0_validi = (data_ultima_utile - d0_inizio).days + 1

st.header("⚙️ Ottimizzazione Economica Stocastica")
if giorni_d0_validi <= 0:
    st.error("🚨 Errore: L'orizzonte di forecast selezionato è troppo corto per permettere il completamento sequenziale dei lavori!")
else:
    st.info(f"⏳ **Configurazione Motore:** Verranno simulate {giorni_d0_validi} date di partenza distinte (D0) ciclando su ciascuno dei {len(forecast_data)} membri ensemble attivi.")


# --- 5. PULSANTE DI OTTIMIZZAZIONE MONTE CARLO ---
if st.button("🚀 Avvia Ottimizzazione Economica", disabled=(giorni_d0_validi <= 0 or not forecast_data)):
    with st.spinner("Estrazione record storici decennali ed elaborazione catene di simulazione stocastiche..."):
        historical_data = fetch_historical_data(lat, lon, d0_inizio)
        
        if historical_data.empty:
            st.error("Errore: I server dell'archivio storico Open-Meteo non rispondono per le coordinate fornite.")
        else:
            intervallo_d0 = [d0_inizio + datetime.timedelta(days=x) for x in range(giorni_d0_validi)]
            risultati_globali = []

            for d0 in intervallo_d0:
                costi_membri = []
                successi_entro_forecast = 0
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

                if costi_membri:
                    risultati_globali.append({
                        "D0": d0, "Costo Medio": np.mean(costi_membri), "P10": np.percentile(costi_membri, 10), "P90": np.percentile(costi_membri, 90),
                        "Prob Successo Forecast %": (successi_entro_forecast / len(costi_membri)) * 100.0, "Richiesto MC %": (usato_monte_carlo / len(costi_membri)) * 100.0
                    })

            df_risultati = pd.DataFrame(risultati_globali)
            if not df_risultati.empty:
                miglior_d0_economico = df_risultati.loc[df_risultati["Costo Medio"].idxmin()]
                st.success(f"🎯 **Giorno d'Inizio Ottimale (Minimo Costo):** **{miglior_d0_economico['D0'].strftime('%d/%m/%Y')}** (Costo Medio Atteso: **{miglior_d0_economico['Costo Medio']:,.2f} €**).")
                
                # Grafico finale dei costi attesi per D0
                fig_costi = go.Figure()
                fig_costi.add_trace(go.Scatter(x=df_risultati["D0"], y=df_risultati["Costo Medio"], mode="lines+markers", name="Costo Medio [€]", line=dict(color="firebrick", width=3)))
                fig_costi.update_layout(title="Curva di Ottimizzazione dei Costi di Cantiere per data di inizio D0", xaxis_title="Giorno d'Inizio (D0)", yaxis_title="Costo Totale Atteso [€]")
                st.plotly_chart(fig_costi, use_container_width=True)
                
                # Tabella riassuntiva finanziaria
                st.subheader("📊 Dettaglio Risultati Finanziari Complessivi")
                df_vis = df_risultati.copy()
                for col in ["Costo Medio", "P10", "P90"]: df_vis[col] = df_vis[col].map("{:,.2f} €".format)
                for col in ["Prob Successo Forecast %", "Richiesto MC %"]: df_vis[col] = df_vis[col].map("{:.1f} %".format)
                st.dataframe(df_vis, use_container_width=True)

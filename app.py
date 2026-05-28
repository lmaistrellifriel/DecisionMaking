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
    "ICON Seamless": {"api_name": "icon_seamless", "max_membri": 40}
}

modello_sel = st.sidebar.selectbox("Modello Ensemble", list(modelli_dict.keys()))
max_m = modelli_dict[modello_sel]["max_membri"]

membri_richiesti = st.sidebar.number_input(
    f"Numero Ensemble (Max {max_m})", 
    min_value=2, 
    max_value=max_m, 
    value=min(10, max_m)
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

# --- 2. DOWNLOAD DATI IN PARALLELO CORRETTO ---
def fetch_forecast_data(lat, lon, model_api, giorni, n_membri, max_m):
    # Logica di Campionamento Alternato perfettamente implementata
    passo = max(1, max_m // n_membri)
    membri_scelti = [i for i in range(0, max_m, passo)][:n_membri]
    
    url = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&models={model_api}&windspeed_100m=true&windgusts_10m=true&forecast_days={giorni}"
    res = requests.get(url).json()
    
    hourly_data = res.get("hourly", {})
    v_key = f"windspeed_100m_{model_api}" if f"windspeed_100m_{model_api}" in hourly_data else f"windspeed_10m_{model_api}"
    g_key = f"windgusts_10m_{model_api}"
    
    time_seq = pd.to_datetime(hourly_data.get("time", []))
    is_kmh = res.get("hourly_units", {}).get(v_key, "km/h") == "km/h"
    
    forecast_ensemble = {}
    for m in membri_scelti:
        v_mem = hourly_data.get(f"{v_key}_member{m}", hourly_data.get(f"windspeed_10m_member{m}", []))
        g_mem = hourly_data.get(f"{g_key}_member{m}", hourly_data.get(f"windgusts_10m_member{m}", []))
        
        if is_kmh:
            v_mem = [v / 3.6 for v in v_mem]
            g_mem = [g / 3.6 for g in g_mem]
            
        forecast_ensemble[m] = pd.DataFrame({"vento": v_mem, "raffica": g_mem}, index=time_seq)
    return forecast_ensemble

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
        res = requests.get(url).json()
        hourly_data = res.get("hourly", {})
        
        v_data = hourly_data.get("windspeed_100m", hourly_data.get("windspeed_10m", []))
        g_data = hourly_data.get("windgusts_10m", [])
        time_seq = pd.to_datetime(hourly_data.get("time", []))
        
        if res.get("hourly_units", {}).get("windspeed_100m", "km/h") == "km/h":
            v_data = [v / 3.6 for v in v_data]
            g_data = [g / 3.6 for g in g_data]
            
        df_anno = pd.DataFrame({"vento": v_data, "raffica": g_data, "anno": anno}, index=time_seq)
        df_storico_lista.append(df_anno)
        
    return pd.concat(df_storico_lista)

# Logica di controllo orizzonte temporale
durata_teorica_ore = df_steps_input["Durata [h]"].sum()
ore_lavorative_giorno = (fine_turno.hour + fine_turno.minute/60) - (inizio_turno.hour + inizio_turno.minute/60)
giorni_teorici_minimi = int(np.ceil(durata_teorica_ore / ore_lavorative_giorno))

max_forecast_date = d0_inizio + datetime.timedelta(days=giorni_forecast)
data_ultima_utile = max_forecast_date - datetime.timedelta(days=giorni_teorici_minimi)
giorni_d0_validi = (data_ultima_utile - d0_inizio).days + 1

if giorni_d0_validi <= 0:
    st.error("🚨 Errore: L'orizzonte di forecast richiesto è troppo corto per permettere il completamento dei lavori!")
else:
    tempo_stimato_cpu = giorni_d0_validi * membri_richiesti * 0.005
    st.info(f"⏳ **Preview Tempi di Calcolo:** Saranno simulati **{giorni_d0_validi} giorni (D0)** su **{membri_richiesti} membri**. Tempo stimato CPU Cloud: ~{tempo_stimato_cpu:.3f} secondi.")

if st.button("🚀 Avvia Ottimizzazione", disabled=(giorni_d0_validi <= 0)):
    with st.spinner("Scaricamento dati Open-Meteo ed elaborazione Monte Carlo stocastica..."):
        with ThreadPoolExecutor() as executor:
            future_forecast = executor.submit(fetch_forecast_data, lat, lon, modelli_dict[modello_sel]["api_name"], giorni_forecast, membri_richiesti, max_m)
            future_historical = executor.submit(fetch_historical_data, lat, lon, d0_inizio)
            
            forecast_data = future_forecast.result()
            historical_data = future_historical.result()

        intervallo_d0 = [d0_inizio + datetime.timedelta(days=x) for x in range(giorni_d0_validi)]
        risultati_globali = []
        profili_orari_wind_prod = {}

        for d0 in intervallo_d0:
            costi_membri = []
            successi_entro_forecast = 0
            successi_totali_con_mc = 0
            usato_monte_carlo = 0
            
            profili_orari_wind_prod[d0] = {"vento": [], "produzione": [], "is_forecast": []}

            for m, df_forecast_m in forecast_data.items():
                start_sim_time = pd.Timestamp(datetime.datetime.combine(d0, datetime.time(0, 0)))
                df_m = df_forecast_m[df_forecast_m.index >= start_sim_time].copy()
                
                step_corrente_idx = 0
                ore_progresso_step = 0
                costo_accumulato = 0.0
                current_time = start_sim_time
                
                is_mc_active = False
                storico_anno_scelto = None
                delta_ore_mc = 0
                
                log_vento_m = []
                log_prod_m = []
                log_is_forecast_m = []

                while step_corrente_idx < len(df_steps_input):
                    if not is_mc_active and current_time in df_m.index:
                        v_ora = df_m.loc[current_time, "vento"]
                        g_ora = df_m.loc[current_time, "raffica"]
                        is_forecast_flag = True
                    else:
                        if not is_mc_active:
                            is_mc_active = True
                            usato_monte_carlo += 1
                            anni_disponibili = historical_data["anno"].unique()
                            storico_anno_scelto = np.random.choice(anni_disponibili)
                        
                        tempo_mc = pd.Timestamp(datetime.datetime.combine(
                            datetime.date(storico_anno_scelto, current_time.month, current_time.day),
                            current_time.time()
                        )) + datetime.timedelta(hours=delta_ore_mc)
                        
                        if tempo_mc not in historical_data.index:
                            tempo_mc = historical_data[historical_data["anno"] == storico_anno_scelto].index[0]
                            delta_ore_mc = 0
                            
                        v_ora = historical_data.loc[tempo_mc, "vento"]
                        g_ora = historical_data.loc[tempo_mc, "windgusts_10m"]
                        is_forecast_flag = False
                        delta_ore_mc += 1
                    
                    log_vento_m.append(v_ora)
                    log_prod_m.append(calcola_mancata_produzione(v_ora))
                    log_is_forecast_m.append(is_forecast_flag)
                    
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
                        if not is_ora_lavorativa:
                            costo_gru_ora = c_notturno
                        else:
                            if (v_ora <= soglia_v) and (g_ora <= soglia_g):
                                costo_gru_ora = c_operativo
                            else:
                                costo_gru_ora = c_standby
                    else:
                        costo_gru_ora = 0.0
                        
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
                
                costo_totale_scenario = costo_accumulato + c_mob
                costi_membri.append(costo_totale_scenario)
                
                if not is_mc_active:
                    successi_entro_forecast += 1
                successi_totali_con_mc += 1
                
                if len(profili_orari_wind_prod[d0]["vento"]) == 0:
                    profili_orari_wind_prod[d0]["vento"] = np.array(log_vento_m)
                    profili_orari_wind_prod[d0]["produzione"] = np.array(log_prod_m)
                    profili_orari_wind_prod[d0]["is_forecast"] = np.array(log_is_forecast_m)
                else:
                    min_len = min(len(profili_orari_wind_prod[d0]["vento"]), len(log_vento_m))
                    profili_orari_wind_prod[d0]["vento"] = (profili_orari_wind_prod[d0]["vento"][:min_len] + np.array(log_vento_m[:min_len])) / 2
                    profili_orari_wind_prod[d0]["produzione"] = (profili_orari_wind_prod[d0]["produzione"][:min_len] + np.array(log_prod_m[:min_len])) / 2
                    profili_orari_wind_prod[d0]["is_forecast"] = profili_orari_wind_prod[d0]["is_forecast"][:min_len]
            
            prob_successo_forecast = (successi_entro_forecast / membri_richiesti) * 100.0
            prob_successo_totale = (successi_totali_con_mc / membri_richiesti) * 100.0
            pct_necessitato_mc = (usato_monte_carlo / membri_richiesti) * 100.0
            
            costo_medio_d0 = np.mean(costi_membri)
            p10_d0 = np.percentile(costi_membri, 10)
            p90_d0 = np.percentile(costi_membri, 90)
            
            risultati_globali.append({
                "D0": d0, "Costo Medio": costo_medio_d0, "P10": p10_d0, "P90": p90_d0, "Spread": p90_d0 - p10_d0,
                "Prob Successo Forecast %": prob_successo_forecast, "Prob Successo Totale %": prob_successo_totale, "Richiesto MC %": pct_necessitato_mc
            })

        df_risultati = pd.DataFrame(risultati_globali)
        miglior_d0_economico = df_risultati.loc[df_risultati["Costo Medio"].idxmin()]
        miglior_d0_sicurezza = df_risultati.loc[df_risultati["Prob Successo Forecast %"].idxmax()]

        # Output Risultati
        st.success(f"🎯 **Giorno d'Inizio Ottimale (Minimo Costo):** **{miglior_d0_economico['D0'].strftime('%d/%m/%Y')}** con Costo Medio Atteso di **{miglior_d0_economico['Costo Medio']:,.2f} €**.")
        st.info(f"🛡️ **Giorno con Massima Sicurezza (Successo in Forecast):** **{miglior_d0_sicurezza['D0'].strftime('%d/%m/%Y')}** ({miglior_d0_sicurezza['Prob Successo Forecast %']:.1f}% di probabilità).")

        # --- GRAFICI INTERATTIVI ---
        st.header("📊 Analisi Grafica e Ottimizzazione")
        fig_costi = go.Figure()
        fig_costi.add_trace(go.Scatter(
            x=df_risultati["D0"], y=df_risultati["Costo Medio"], mode="lines+markers", name="Costo Medio [€]",
            line=dict(color="firebrick", width=3),
            error_y=dict(type="data", symmetric=False, array=df_risultati["P90"] - df_risultati["Costo Medio"], arrayminus=df_risultati["Costo Medio"] - df_risultati["P10"], visible=True, color="rgba(240, 50, 50, 0.3)"),
            hovertemplate="<b>Data D0: %{x}</b><br>Costo Medio: %{y:,.2f} €<br>P10: %{customdata[0]:,.2f} €<br>P90: %{customdata[1]:,.2f} €<br>Prob. Forecast: %{customdata[3]:.1f}%<extra></extra>",
            customdata=np.stack((df_risultati["P10"], df_risultati["P90"], df_risultati["Spread"], df_risultati["Prob Successo Forecast %"]), axis=-1)
        ))
        st.plotly_chart(fig_costi, use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            d0_opt = miglior_d0_economico["D0"]
            profilo_opt = profili_orari_wind_prod[d0_opt]
            fig_meteo = go.Figure()
            fig_meteo.add_trace(go.Scatter(x=[i for i in range(len(profilo_opt["vento"]))], y=profilo_opt["vento"], mode="lines", name="Vento Medio [m/s]", line=dict(color="navy", width=2)))
            idx_cambio = np.where(profilo_opt["is_forecast"] == False)[0]
            if len(idx_cambio) > 0:
                fig_meteo.add_vline(x=idx_cambio[0], line_dash="dash", line_color="orange", annotation_text="Inizio Monte Carlo")
            st.plotly_chart(fig_meteo, use_container_width=True)

        with col_g2:
            fig_pc = go.Figure()
            fig_pc.add_trace(go.Scatter(x=v_curve, y=p_curve, mode="lines", name="Power Curve", line=dict(color="green", width=2)))
            st.plotly_chart(fig_pc, use_container_width=True)

        st.subheader("📊 Dettaglio Dati Finanziari")
        df_vis = df_risultati.copy()
        for col in ["Costo Medio", "P10", "P90", "Spread"]: df_vis[col] = df_vis[col].map("{:,.2f} €".format)
        for col in ["Prob Successo Forecast %", "Prob Successo Totale %", "Richiesto MC %"]: df_vis[col] = df_vis[col].map("{:.1f} %".format)
        st.dataframe(df_vis, use_container_width=True)

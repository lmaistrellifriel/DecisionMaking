# ... (parte iniziale invariata fino alla sezione sidebar)

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
    
    # Selezione ridotta ai 3 modelli principali con i parametri API corretti
    model = st.selectbox(
        "Modello", 
        options=[
            "gfs_seamless", 
            "icon_seamless", 
            "ecmwf_ifs025_ensemble"
        ],
        index=2
    )
    
    forecast_days = st.slider("Giorni forecast (richiesti)", 3, 16, 10)
    include_gusts = st.toggle("Usa raffiche", value=True)

# ... (resto del codice)

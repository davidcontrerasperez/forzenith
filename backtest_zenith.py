import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from strategy_zenith import generate_signal, tracker
from utils import load_sample_data
import random
import os

st.set_page_config(page_title="Zenith Dashboard", layout="wide")
st.title("🧠 Estrategia Zenith: Evolución de Pesos y Rendimiento")

# Parámetros de costes
COSTE_FIJO = 0.2
SPREAD = 0.0003  # 3 pips

# Cargar estado anterior si existe
if os.path.exists("tracker_state.json"):
    tracker.load_from_disk("tracker_state.json")

if "history" not in st.session_state:
    st.session_state.history = tracker.history.copy()

if st.button("🔁 Ejecutar nueva iteración"):
    df = load_sample_data()
    signal = generate_signal(df)
    st.success(f"Señal generada: **{signal}**")

    precio = df['close'].iloc[-1]
    coste_proporcional = SPREAD * precio * 100_000 / 1000  # coste en unidades de PnL

    for name in tracker.weights:
        pnl_bruto = random.uniform(-1.5, 2.5)
        coste_total = COSTE_FIJO + coste_proporcional
        tracker.update(name, pnl_bruto, coste_total)

    tracker.record_weights()
    tracker.save_to_disk("tracker_state.json")
    st.session_state.history.append(tracker.weights.copy())

# Mostrar gráfico de pesos
if st.session_state.history:
    df_weights = pd.DataFrame(st.session_state.history)
    st.subheader("⚖️ Evolución de pesos por agente")
    st.line_chart(df_weights)

# Mostrar PnL acumulado
if tracker.pnl_history:
    pnl_bruto_df = pd.DataFrame(tracker.pnl_history).cumsum()
    pnl_neto_df = pd.DataFrame(tracker.pnl_net_history).cumsum()

    st.subheader("📊 Rendimiento acumulado por agente")
    st.markdown("**Bruto** (sin costes):")
    st.line_chart(pnl_bruto_df)
    st.markdown("**Neto** (con costes):")
    st.line_chart(pnl_neto_df)

# Botón para reiniciar
if st.button("🧨 Reiniciar estado"):
    if os.path.exists("tracker_state.json"):
        os.remove("tracker_state.json")
    st.session_state.history = []
    st.success("Estado reiniciado. Refresca la página para comenzar de nuevo.")
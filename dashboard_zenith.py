import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import signal
import psutil
import json
import numpy as np
import yfinance as yf
from agentes_salida import agentes_salida

# Función para graficar DataFrames de forma segura
def graficar_line_chart(df, titulo=""):
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df = df.select_dtypes(include=[np.number])
        df = df.loc[:, df.applymap(lambda x: isinstance(x, (int, float, np.number))).all()]
        if not df.empty:
            if titulo:
                st.subheader(titulo)
            st.line_chart(df)
        else:
            st.info("No hay datos numéricos válidos para graficar.")
    else:
        st.warning("El objeto proporcionado no es un DataFrame.")

def normalizar_diccionario(diccionario):
    max_len = max(len(v) for v in diccionario.values())
    return {
        k: v + [0.0] * (max_len - len(v))
        for k, v in diccionario.items()
    }

from datetime import datetime, timedelta

from tracker import Tracker
from performance_tracker import PerformanceTracker
from coordinator import Coordinator
from agentes_zenith import AGENTES
from strategy_zenith import generate_signal, tracker

# Configuración de página
st.set_page_config(page_title="Zenith Dashboard", layout="wide")
st.title("🧠 Estrategia Zenith: Evolución de Pesos y Rendimiento")

def actualizar_pesos_salida(resultados, pesos_actuales, normalizar=True):
    nuevo_pesos = pesos_actuales.copy()
    suma_abs = sum(abs(v) for v in resultados.values())

    if suma_abs == 0:
        return nuevo_pesos

    for agente, resultado in resultados.items():
        incremento = (resultado / suma_abs) * 5
        nuevo_pesos[agente] += incremento

    if normalizar:
        total = sum(nuevo_pesos.values())
        if total > 0:
            for agente in nuevo_pesos:
                nuevo_pesos[agente] /= total

    return nuevo_pesos

def evaluar_agentes_con_datos_reales(sim_tracker, sim_perf_tracker, sim_coordinator, simbolo="EURUSD=X"):
    import yfinance as yf
    import numpy as np

    df = yf.download(simbolo, period="6mo", interval="1d")
    df.dropna(inplace=True)

    firma_historial = {}
    capital_total = 1000

    pesos_salida = {nombre: 1/len(agentes_salida) for nombre in agentes_salida}
    operaciones_abiertas = []

    for i in range(30, len(df)):
        df_slice = df.iloc[:i+1].copy()
        df_slice.rename(columns=str.lower, inplace=True)
        fecha_operacion = df_slice.index[-1]
        precio_actual = df_slice['close'].iloc[-1]
        coste = 0.05

        # Cierre de salidas
        nuevas_operaciones_abiertas = []
        for operacion in operaciones_abiertas:
            resultados_salida = {}
            for salida in operacion["salidas"]:
                if not salida["cerrada"]:
                    try:
                        decision_salida = agentes_salida[salida["nombre"]](df_slice)
                        if bool(decision_salida):
                            salida["cerrada"] = True
                            salida["precio_salida"] = precio_actual
                            salida["resultado"] = (precio_actual - operacion["precio_entrada"]) / operacion["precio_entrada"] * salida["capital"]
                            resultados_salida[salida["nombre"]] = salida["resultado"]

                            sim_perf_tracker.update(
                                f"{operacion['agente_entrada']}_{salida['nombre']}",
                                salida["resultado"],
                                coste / len(agentes_salida),
                                fecha=fecha_operacion,
                                simbolo=simbolo,
                                peso_actual=salida["capital"] / capital_total,
                                capital_total=salida["capital"]
                            )
                    except Exception as e:
                        print(f"Error en agente de salida {salida['nombre']}: {e}")
                    
            if all(s["cerrada"] for s in operacion["salidas"]):
                pesos_salida = actualizar_pesos_salida(resultados_salida, pesos_salida)
            else:
                nuevas_operaciones_abiertas.append(operacion)
        operaciones_abiertas = nuevas_operaciones_abiertas

        # Entrada
        signal = sim_coordinator.decide(
            df_slice,
            iteracion=sim_tracker.iteracion,
            historial=sim_tracker.historial,
            memoria=sim_tracker.memoria,
            df_1h=df_slice,
            df_4h=df_slice,
            eventos=[],
            fase_actual="llena"
        )

        decisiones = sim_coordinator.last_decisions
        firma = tuple(decisiones[nombre] for nombre in sorted(decisiones))
        hist = firma_historial.get(firma, {"ganadas": 0, "perdidas": 0})

        multiplicador = 1.0
        if hist["ganadas"] > 0:
            multiplicador *= hist["ganadas"]
        if hist["perdidas"] > 0:
            multiplicador /= hist["perdidas"]
        multiplicador = max(multiplicador, 0.1)

        pesos_operativos = {
            nombre: sim_tracker.weights.get(nombre, 0.0)
            for nombre, decision in decisiones.items()
            if decision != "hold"
        }
        total_pesos = sum(pesos_operativos.values())

        ciclo_resultado = 0.0

        for nombre, decision in decisiones.items():
            if decision != "hold":
                if nombre in pesos_operativos and total_pesos > 0:
                    peso_relativo = pesos_operativos[nombre] / total_pesos
                else:
                    peso_relativo = 0.0

                capital_ajustado = capital_total * peso_relativo * multiplicador

                operacion = {
                    "agente_entrada": nombre,
                    "capital": capital_ajustado,
                    "fecha_entrada": fecha_operacion,
                    "precio_entrada": precio_actual,
                    "salidas": [
                        {
                            "nombre": salida,
                            "capital": capital_ajustado * pesos_salida[salida],
                            "cerrada": False
                        }
                        for salida in agentes_salida
                    ]
                }
                operaciones_abiertas.append(operacion)

        if ciclo_resultado > 0:
            hist["ganadas"] += 1
        else:
            hist["perdidas"] += 1
        firma_historial[firma] = hist

        sim_tracker.ajustar_pesos_por_rendimiento(decisiones, ciclo_resultado)
        sim_perf_tracker.record_weights()

    return pd.DataFrame(sim_perf_tracker.history), sim_perf_tracker, firma_historial

if st.button("📡 Evaluar agentes con datos reales (últimos 6 meses)"):
    sim_tracker = Tracker("EURUSD")
    sim_perf_tracker = PerformanceTracker()
    sim_coordinator = Coordinator(sim_tracker)
    for nombre in AGENTES:
        sim_perf_tracker.register_agent(nombre)

    df_pesos, sim_perf_tracker, firma_historial = evaluar_agentes_con_datos_reales(sim_tracker, sim_perf_tracker, sim_coordinator)

    st.success("Evaluación completada con datos reales.")

    graficar_line_chart(df_pesos, "📊 Evolución de pesos con datos reales")

    pnl_bruto_df = pd.DataFrame(normalizar_diccionario(sim_perf_tracker.pnl_history)).cumsum()
    pnl_neto_df = pd.DataFrame(normalizar_diccionario(sim_perf_tracker.pnl_net_history)).cumsum()

    st.subheader("📈 Rendimiento con datos reales")
    st.markdown("**Bruto (sin costes):**")
    graficar_line_chart(pnl_bruto_df)
    st.markdown("**Neto (con costes):**")
    graficar_line_chart(pnl_neto_df)

    df_net = pd.DataFrame(normalizar_diccionario(sim_perf_tracker.pnl_net_history))
    pnl_totales = df_net.sum().sort_values(ascending=False)
    df_ranking = pnl_totales.reset_index()
    df_ranking.columns = ["Agente", "PnL Neto Acumulado"]
    st.subheader("🏆 Ranking con datos reales")
    st.dataframe(df_ranking.style.format({"PnL Neto Acumulado": "{:.2f}"}))
    st.bar_chart(df_ranking.set_index("Agente"))

# 💰 Resultado total de inversión simulada
    capital_inicial = 1000
    pnl_total = sum([sum(v) for v in sim_perf_tracker.pnl_net_history.values()])
    capital_final = capital_inicial + pnl_total

    st.subheader("💰 Resultado total de inversión")
    st.markdown(f"Si hubieras invertido **1000 €** aplicando todos los agentes durante los últimos 6 meses, ahora tendrías     aproximadamente **{capital_final:.2f} €**.")
# 💸 Resultado de inversión simulada por agente
    st.subheader("💸 Resultado de inversión simulada por agente")

    capital_inicial = 1000
    agentes = list(sim_perf_tracker.pnl_net_history.keys())
    n_agentes = len(agentes)
    capital_por_agente = capital_inicial / n_agentes

    resultados = {}

    for agente in agentes:
        pnl_agente = sum(sim_perf_tracker.pnl_net_history[agente])
        capital_final_agente = capital_por_agente + pnl_agente
        resultados[agente] = capital_final_agente

    df_resultados = pd.DataFrame.from_dict(resultados, orient="index", columns=["Capital Final (€)"])
    df_resultados = df_resultados.sort_values("Capital Final (€)", ascending=False)

    st.dataframe(df_resultados.style.format({"Capital Final (€)": "€{:.2f}"}))
    st.bar_chart(df_resultados)

# Mostrar operaciones realizadas
    df_operaciones = pd.DataFrame(sim_perf_tracker.operaciones)

    if not df_operaciones.empty:
        capital_total = 1000
        n_agentes = len(AGENTES)
        capital_por_agente = capital_total / n_agentes

        df_operaciones["rendimiento_%"] = 100 * df_operaciones["pnl_neto"] / capital_por_agente
        df_operaciones["dinero_invertido"] = df_operaciones["capital_invertido"]
        df_operaciones["rendimiento_%"] = 100 * df_operaciones["pnl_neto"] / df_operaciones["dinero_invertido"]
        df_operaciones["fecha"] = pd.to_datetime(df_operaciones["fecha"])

        st.subheader("📋 Historial de operaciones (últimos 6 meses)")
        st.dataframe(
            df_operaciones[[
                "fecha", "agente", "moneda", "pnl_neto", "rendimiento_%", "dinero_invertido"
            ]].sort_values("fecha").reset_index(drop=True).style.format({
                "pnl_neto": "€{:.2f}",
                "rendimiento_%": "{:.2f}%",
                "dinero_invertido": "€{:.2f}"
            })
        )
# 📊 Resumen de rendimiento por agente
        st.subheader("📈 Resumen de rendimiento por agente")

# Agrupar operaciones por agente
        resumen = df_operaciones.groupby("agente").agg(
            operaciones_totales=("pnl_neto", "count"),
            operaciones_positivas=("pnl_neto", lambda x: (x > 0).sum()),
            pnl_total=("pnl_neto", "sum")
        )

        resumen["tasa_acierto_%"] = 100 * resumen["operaciones_positivas"] / resumen["operaciones_totales"]

# Ordenar por tasa de acierto
        resumen = resumen.sort_values("tasa_acierto_%", ascending=False)

# Mostrar tabla
        st.dataframe(resumen.style.format({
            "pnl_total": "€{:.2f}",
            "tasa_acierto_%": "{:.2f}%",
            "operaciones_totales": "{:.0f}",
            "operaciones_positivas": "{:.0f}"
        }))
# 📊 Ranking de combinaciones de decisiones (firmas)
        st.subheader("🔐 Ranking de combinaciones de decisiones")

# Convertir el historial de firmas a DataFrame
        df_firmas = pd.DataFrame([
            {
                "firma": str(firma),
                "veces_usada": hist["ganadas"] + hist["perdidas"],
                "ganadas": hist["ganadas"],
                "perdidas": hist["perdidas"],
                "tasa_acierto_%": 100 * hist["ganadas"] / (hist["ganadas"] + hist["perdidas"]) if (hist["ganadas"] + hist["perdidas"]) > 0 else 0
            }
            for firma, hist in firma_historial.items()
        ])

# Ordenar por tasa de acierto
        df_firmas = df_firmas.sort_values("tasa_acierto_%", ascending=False)

# Mostrar tabla
        st.dataframe(df_firmas.style.format({
            "tasa_acierto_%": "{:.2f}%",
            "veces_usada": "{:.0f}",
            "ganadas": "{:.0f}",
            "perdidas": "{:.0f}"
        }))

st.title("🧠 Estrategia Zenith: Evolución de Pesos y Rendimiento")

def reiniciar_agentes():
    tracker.weights = {nombre: 1.0 for nombre in AGENTES}
    total = sum(tracker.weights.values())
    for k in tracker.weights:
        tracker.weights[k] /= total

    tracker.history = []
    tracker.pnl_history = {}
    tracker.pnl_net_history = {}

    st.session_state.history = []
    st.session_state.current_position = "Flat"
    st.session_state.pnl_total = 0.0
if st.button("🔄 Reiniciar agentes"):
    reiniciar_agentes()
    st.success("Agentes reiniciados correctamente.")

# Archivo de estado persistente
STATE_FILE = "bot_state.json"

def guardar_estado_bot(activo):
    estado = {"activo": activo}
    with open(STATE_FILE, "w") as f:
        json.dump(estado, f)

def cargar_estado_bot():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            estado = json.load(f)
        return estado.get("activo", False)
    return False

# Cargar estado persistente
bot_activo = cargar_estado_bot()

# Panel lateral
st.sidebar.title("⚙️ Estado del bot Zenith")

if not bot_activo:
    st.sidebar.markdown("### 🔴 Estado: **Inactivo**")
    if st.sidebar.button("▶️ Activar bot"):
        process = subprocess.Popen(["python", "zenith_bot.py"])
        guardar_estado_bot(True)
        st.rerun()
else:
    st.sidebar.markdown("### 🟢 Estado: **Activo**")
    if st.sidebar.button("⏹️ Detener bot"):
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if "zenith_bot.py" in " ".join(proc.info["cmdline"]):
                    proc.terminate()
            guardar_estado_bot(False)
            st.rerun()
        except Exception as e:
            guardar_estado_bot(False)
            st.sidebar.warning("Error al detener el bot. Estado limpiado.")
            st.rerun()

# Inicializar estado del tracker
if os.path.exists("tracker_state.json"):
    tracker.load_from_disk("tracker_state.json")

# Crear objetos compartidos para simulación
sim_tracker = Tracker("EURUSD")
sim_perf_tracker = PerformanceTracker()
sim_coordinator = Coordinator(sim_tracker)
for nombre in AGENTES:
    sim_perf_tracker.register_agent(nombre)

# Simulación
st.subheader("🧪 Simulación de estrategia")

if st.button("Ejecutar simulación de 100 ciclos"):
    equity, df_pesos = simular_ciclos(sim_tracker, sim_perf_tracker, sim_coordinator)

    if equity:
        st.line_chart(equity, height=300, use_container_width=True)
        st.caption("📈 Evolución del capital (equity curve)")

    if not df_pesos.empty:
        graficar_line_chart(df_pesos, "📊 Evolución de los pesos de los agentes")

    st.session_state.history = sim_tracker.history
    st.session_state.pnl_total = equity[-1] - equity[0]
    st.session_state.current_position = "Simulado"
    tracker.weights = sim_tracker.weights
    tracker.history = sim_tracker.history
    tracker.pnl_history = sim_perf_tracker.pnl_history
    tracker.pnl_net_history = sim_perf_tracker.pnl_net_history

# Estado inicial
if "history" not in st.session_state:
    st.session_state.history = tracker.history.copy()
if "current_position" not in st.session_state:
    st.session_state.current_position = "Flat"
if "pnl_total" not in st.session_state:
    st.session_state.pnl_total = 0.0

# Mostrar posición actual y PnL acumulado
st.markdown(f"### 📍 Posición actual: `{st.session_state.current_position}`")
st.markdown(f"### 💰 PnL acumulado: `{round(st.session_state.pnl_total, 2)} unidades`")

# 📊 Gráfico de pesos
if st.session_state.history:
    df_weights = pd.DataFrame(st.session_state.history)
    graficar_line_chart(df_weights, "⚖️ Evolución de pesos por agente")

# 📈 Gráfico de rendimiento
if tracker.pnl_history:
    pnl_bruto_df = pd.DataFrame(normalizar_diccionario(tracker.pnl_history)).cumsum()
    pnl_neto_df = pd.DataFrame(normalizar_diccionario(tracker.pnl_net_history)).cumsum()

    st.subheader("📊 Rendimiento acumulado por agente")
    st.markdown("**Bruto (sin costes):**")
    graficar_line_chart(pnl_bruto_df)
    st.markdown("**Neto (con costes):**")
    graficar_line_chart(pnl_neto_df)

# 🏆 Ranking de agentes
if tracker.pnl_net_history:
    df_net = pd.DataFrame(normalizar_diccionario(tracker.pnl_net_history))
    pnl_totales = df_net.sum().sort_values(ascending=False)
    df_ranking = pnl_totales.reset_index()
    df_ranking.columns = ["Agente", "PnL Neto Acumulado"]
    st.subheader("🏆 Ranking de agentes por rendimiento neto acumulado")
    st.dataframe(df_ranking.style.format({"PnL Neto Acumulado": "{:.2f}"}))
    st.bar_chart(df_ranking.set_index("Agente"))

# 📄 Log de acciones del bot
st.subheader("📄 Registro de actividad del bot")
log_file = "zenith_log.txt"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        log_lines = f.readlines()
    st.text_area("Log del bot", value="".join(log_lines[-100:]), height=300)
else:
    st.info("Aún no hay log registrado.")

# ⚖️ Pesos actuales de los agentes
st.subheader("⚖️ Pesos actuales de los agentes")
if tracker.weights:
    df_pesos = pd.DataFrame.from_dict(tracker.weights, orient="index", columns=["Peso"])
    df_pesos = df_pesos.sort_values("Peso", ascending=False)
    st.dataframe(df_pesos.style.format({"Peso": "{:.2%}"}))
    st.bar_chart(df_pesos)
else:
    st.info("No hay pesos registrados aún.")

# 📉 Evolución histórica de los pesos
st.subheader("📉 Evolución histórica de los pesos")
if tracker.history:
    df_hist = pd.DataFrame(tracker.history)
    if "timestamp" in df_hist.columns:
        df_hist = df_hist.drop(columns=["timestamp"])
    df_hist = df_hist.select_dtypes(include=[np.number])
    if not df_hist.empty:
        graficar_line_chart(df_hist, "📉 Evolución histórica de los pesos")
    else:
        st.info("No hay datos numéricos válidos para graficar.")
else:
    st.info("Aún no hay historial de pesos registrado.")

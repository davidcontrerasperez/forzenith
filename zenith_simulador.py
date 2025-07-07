import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from tracker import Tracker
from performance_tracker import PerformanceTracker
from coordinator import Coordinator
from agentes_zenith import AGENTES

# ============================
# Función de simulación de datos
# ============================

def get_data(symbol, timeframe="1h", n=100):
    now = datetime.now()
    delta = {"1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)}.get(timeframe, timedelta(hours=1))
    dates = [now - i * delta for i in range(n)][::-1]

    prices = np.cumsum(np.random.randn(n)) + 100
    highs = prices + np.random.rand(n)
    lows = prices - np.random.rand(n)
    opens = prices + np.random.randn(n) * 0.2
    closes = prices + np.random.randn(n) * 0.2
    volumes = np.random.randint(100, 1000, size=n)

    df = pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    df.set_index("datetime", inplace=True)
    return df

# ============================
# Simulación de múltiples ciclos
# ============================

symbol = "EURUSD"
tracker = Tracker(symbol)
performance_tracker = PerformanceTracker()
coordinator = Coordinator(tracker)

# Registrar todos los agentes
for nombre in AGENTES:
    performance_tracker.register_agent(nombre)

equity = [10000]  # capital inicial
capital = 10000

for ciclo in range(100):  # número de ciclos
    df = get_data(symbol)
    df_1h = get_data(symbol, timeframe="1h")
    df_4h = get_data(symbol, timeframe="4h")
    eventos = [{"nombre": "Tipos de interés", "fecha": datetime.now().strftime("%Y-%m-%d"), "impacto": "alto"}]

    signal = coordinator.decide(
        df,
        iteracion=tracker.iteracion,
        historial=tracker.historial,
        memoria=tracker.memoria,
        df_1h=df_1h,
        df_4h=df_4h,
        eventos=eventos,
        fase_actual="nueva"
    )

    resultado = np.random.uniform(-1.0, 1.0) if signal != "hold" else 0.0
    coste = 0.1
    capital += resultado - coste
    equity.append(capital)

    tracker.ajustar_pesos_por_rendimiento(coordinator.last_decisions, resultado)
    for nombre, decision in coordinator.last_decisions.items():
        if decision != "hold":
            performance_tracker.update(nombre, resultado, coste)

    performance_tracker.record_weights()

# ============================
# Visualización de resultados
# ============================

# 📈 Equity curve
plt.figure(figsize=(10, 4))
plt.plot(equity, label="Equity")
plt.title("Evolución del capital")
plt.xlabel("Ciclo")
plt.ylabel("Capital")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 📊 Evolución de pesos
df_pesos = pd.DataFrame(performance_tracker.history)
if not df_pesos.empty:
    df_pesos.plot(figsize=(12, 6), title="Evolución de pesos por agente")
    plt.xlabel("Ciclo")
    plt.ylabel("Peso")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tracker import Tracker
from performance_tracker import PerformanceTracker
from coordinator import Coordinator
from agentes_zenith import AGENTES

# ============================
# Función de simulación de datos
# ============================

def get_data(symbol, timeframe="1h", n=100):
    """
    Simula datos OHLCV para un símbolo y timeframe dado.
    """
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
# Ciclo de decisión del bot
# ============================

symbol = "EURUSD"
tracker = Tracker(symbol)
performance_tracker = PerformanceTracker()
coordinator = Coordinator(tracker)

# Registrar todos los agentes en el PerformanceTracker
for nombre in AGENTES:
    performance_tracker.register_agent(nombre)

# Simulación de datos
df = get_data(symbol)
df_1h = get_data(symbol, timeframe="1h")
df_4h = get_data(symbol, timeframe="4h")
eventos = [{"nombre": "Tipos de interés", "fecha": datetime.now().strftime("%Y-%m-%d"), "impacto": "alto"}]

# Decisión del bot
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

# Simulación de resultado
resultado = np.random.uniform(-1.0, 1.0) if signal != "hold" else 0.0
coste_operacion = 0.1

# Ajuste de pesos y rendimiento
tracker.ajustar_pesos_por_rendimiento(coordinator.last_decisions, resultado)
tracker.guardar()

for nombre, decision in coordinator.last_decisions.items():
    if decision != "hold":
        performance_tracker.update(nombre, pnl_bruto=resultado, coste=coste_operacion)

performance_tracker.record_weights()
performance_tracker.save_to_disk()

print(f"📌 Señal final: {signal}")
print(f"💰 Resultado simulado: {resultado:.2f}")
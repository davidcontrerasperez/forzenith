import time
import random
import os
import json
from datetime import datetime

from strategy_zenith import generate_signal, Tracker
from ib_connection import connect_ib, get_latest_data, execute_order
from agentes_zenith import AGENTES

# Configuración
ENTORNO = "paper"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
INTERVALO_SEGUNDOS = 60
COSTE_FIJO = 0.2
SPREAD = 0.0003

# Conectar a IBKR
connect_ib("demo")

# Inicializar trackers por símbolo
trackers = {}
for symbol in SYMBOLS:
    tracker = Tracker()
    state_file = f"tracker_{symbol}.json"
    if os.path.exists(state_file):
        tracker.load_from_disk(state_file)
    trackers[symbol] = tracker

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("zenith_log.txt", "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

log("Zenith bot multimoneda iniciado")

while True:
    for symbol in SYMBOLS:
        try:
            df = get_latest_data(symbol)
            signal = generate_signal(df)
            decisiones = coordinator.last_decisions
            signal = generate_signal(df, 
                iteracion=tracker.iteracion,
                historial=tracker.historial,
                memoria=tracker.memoria,
                df_1h=df_1h,
                df_4h=df_4h,
                eventos=eventos,
                fase_actual="nueva"
            )
            precio = df['close'].iloc[-1]
            coste_proporcional = SPREAD * precio * 100_000 / 1000
            coste_total = COSTE_FIJO + coste_proporcional

            tracker = trackers[symbol]
            pnl_total_iteracion = 0
            for name in tracker.weights:
                pnl_bruto = random.uniform(-1.5, 2.5)
                tracker.update(name, pnl_bruto, coste_total)
                pnl_total_iteracion += pnl_bruto - coste_total

            tracker.ajustar_pesos_por_rendimiento(min_peso=0.05, max_peso=0.6, tasa_aprendizaje=0.3)
            tracker.record_weights()
            tracker.save_to_disk(f"tracker_{symbol}.json")

            log(f"{symbol} → Señal: {signal} | PnL neto: {round(pnl_total_iteracion, 2)}")

            if signal == "buy":
                status = execute_order(symbol, "BUY", 100000)
                log(f"{symbol} → Orden BUY ejecutada — Estado: {status}")
            elif signal == "sell":
                status = execute_order(symbol, "SELL", 100000)
                log(f"{symbol} → Orden SELL ejecutada — Estado: {status}")

        except Exception as e:
            log(f"{symbol} → ❌ Error: {e}")

    time.sleep(INTERVALO_SEGUNDOS)
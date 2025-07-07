import time
import pandas as pd
import random
from strategy_zenith import generate_signal, tracker
from utils import load_sample_data

def get_live_data():
    # Simulación de una nueva vela
    df = load_sample_data()
    new_row = {
        'open': random.uniform(1.05, 1.10),
        'high': random.uniform(1.10, 1.12),
        'low': random.uniform(1.04, 1.08),
        'close': random.uniform(1.06, 1.11),
        'volume': random.randint(100, 1000)
    }
    df = df.append(new_row, ignore_index=True)
    return df.tail(100)

def run_live_mode(interval=60):
    print("🟢 Modo en vivo iniciado (simulado). Presiona Ctrl+C para detener.")
    while True:
        df = get_live_data()
        signal = generate_signal(df)
        print(f"📈 Señal generada: {signal}")

        # Simular PnL aleatorio
        for name in tracker.weights:
            pnl = random.uniform(-1.0, 2.0)
            tracker.update(name, pnl)
        tracker.record_weights()
        tracker.save_to_disk("tracker_state.json")

        time.sleep(interval)

if __name__ == "__main__":
    run_live_mode(interval=60)  # Ejecuta cada 60 segundos
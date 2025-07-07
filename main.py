from strategy_zenith import generate_signal
from utils import load_sample_data
from strategy_zenith import tracker  # Asegúrate de exponer el tracker globalmente
from agentes_zenith import AGENTES
import matplotlib.pyplot as plt
import pandas as pd

# Simular 30 iteraciones
for i in range(30):
    df = load_sample_data()
    signal = generate_signal(df)
    print(f"[{i+1}] Señal: {signal}")
    # Simular PnL aleatorio para cada agente
    for name in tracker.weights:
        import random
        pnl = random.uniform(-1.5, 2.5)
        tracker.update(name, pnl)
    tracker.record_weights()

# Exportar y graficar
tracker.export_history()

df = pd.read_csv("weights_history.csv")
df.plot(title="Evolución de pesos por agente", figsize=(10, 6))
plt.xlabel("Iteración")
plt.ylabel("Peso")
plt.grid(True)
plt.tight_layout()
plt.show()
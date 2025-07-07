from agentes_zenith import AGENTES

from coordinator import Coordinator
from performance_tracker import PerformanceTracker

# 🔁 Tracker global accesible desde otros módulos
tracker = PerformanceTracker()

# Registrar agentes una sola vez
for name, agent in AGENTES.items():
    tracker.register_agent(agent)

# Coordinador global
coordinator = Coordinator(tracker)

# Función principal que genera la señal
def generate_signal(df, **kwargs):
    return coordinator.decide(df, **kwargs)

#para que se ajusten pesos automaticamente
def ajustar_pesos_por_rendimiento(self, min_peso=0.05, max_peso=0.6, tasa_aprendizaje=0.3):
    if not self.pnl_net_history:
        return

    df = pd.DataFrame(self.pnl_net_history)
    rendimiento = df.sum()

    # Evitar valores negativos o nulos
    rendimiento = rendimiento.clip(lower=0.0001)

    # Normalizar
    total = rendimiento.sum()
    pesos_objetivo = (rendimiento / total).to_dict()

    # Aplicar tasa de aprendizaje y limitar pesos
    nuevos_pesos = {}
    for agente in self.weights:
        actual = self.weights[agente]
        objetivo = pesos_objetivo.get(agente, 0)
        ajustado = (1 - tasa_aprendizaje) * actual + tasa_aprendizaje * objetivo
        limitado = min(max(ajustado, min_peso), max_peso)
        nuevos_pesos[agente] = limitado

    # Re-normalizar para que sumen 1
    total_final = sum(nuevos_pesos.values())
    self.weights = {k: v / total_final for k, v in nuevos_pesos.items()}
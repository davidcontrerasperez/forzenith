import json
import os
from datetime import datetime
from agentes_zenith import AGENTES

class Tracker:
    def __init__(self, symbol, path="trackers"):
        self.symbol = symbol
        self.path = path
        self.file = os.path.join(path, f"tracker_{symbol}.json")
        self.weights = {}
        self.history = []
        self.last_decisions = {}
        self.iteracion = 0
        self.historial = []
        self.memoria = {}

        self._load_or_initialize()

    def _load_or_initialize(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if os.path.exists(self.file):
            with open(self.file, "r") as f:
                data = json.load(f)
                self.weights = data.get("weights", {})
                self.history = data.get("history", [])
                self.memoria = data.get("memoria", {})
        else:
            self.weights = {name: 1.0 / len(AGENTES) for name in AGENTES}

        # Asegura que todos los agentes estén sincronizados
        for name in AGENTES:
            if name not in self.weights:
                self.weights[name] = 1.0 / len(AGENTES)

    def get_weight(self, agent_name):
        return self.weights.get(agent_name, 1.0 / len(AGENTES))

    def ajustar_pesos_por_rendimiento(self, decisiones, resultado):
        """
        decisiones: dict con decisiones de cada agente
        resultado: float (PnL o score de la operación)
        """
        for nombre, decision in decisiones.items():
            if resultado > 0:
                self.weights[nombre] *= 1.05  # recompensa
            elif resultado < 0:
                self.weights[nombre] *= 0.95 # castigo

        # Normaliza los pesos
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

        # Guarda en historial
        self.history.append({**self.weights, "timestamp": datetime.now().isoformat()})
        self.iteracion += 1
        self.historial.append({"decision": decisiones, "pnl": resultado})

    def guardar(self):
        with open(self.file, "w") as f:
            json.dump({
                "weights": self.weights,
                "history": self.history,
                "memoria": self.memoria
            }, f, indent=2)
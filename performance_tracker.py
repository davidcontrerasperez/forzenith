import pandas as pd
import os
import json
from datetime import datetime  # Asegúrate de tener esto importado

class PerformanceTracker:
    def __init__(self):
        self.performance = {}
        self.weights = {}
        self.history = []
        self.pnl_history = {}
        self.pnl_net_history = {}
        self.operaciones = []

    def register_agent(self, name):
        self.performance[name] = []
        self.weights[name] = 1.0
        self.pnl_history[name] = []
        self.pnl_net_history[name] = []

    def update(self, nombre, pnl_bruto, coste, fecha=None, simbolo="EURUSD", peso_actual=0.0, capital_total=1000):
        if nombre not in self.pnl_history:
            self.pnl_history[nombre] = []
            self.pnl_net_history[nombre] = []
 
        self.pnl_history[nombre].append(pnl_bruto)
        self.pnl_net_history[nombre].append(pnl_bruto - coste)

        capital_invertido = capital_total * peso_actual

        self.operaciones.append({
            "fecha": fecha or datetime.now(),
            "agente": nombre,
            "moneda": simbolo,
            "pnl_bruto": pnl_bruto,
            "coste": coste,
            "pnl_neto": pnl_bruto - coste,
            "peso": peso_actual,
            "capital_invertido": capital_invertido
        })

    def get_weight(self, name):
        return self.weights.get(name, 1.0)

    def record_weights(self):
        snapshot = {name: weight for name, weight in self.weights.items()}
        self.history.append(snapshot)

    def export_history(self, filename="weights_history.csv"):
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)

    def save_to_disk(self, path="tracker_state.json"):
        state = {
            "performance": self.performance,
            "weights": self.weights,
            "history": self.history,
            "pnl_history": self.pnl_history,
            "pnl_net_history": self.pnl_net_history
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load_from_disk(self, path="tracker_state.json"):
        if os.path.exists(path):
            with open(path, "r") as f:
                state = json.load(f)
                self.performance = state.get("performance", {})
                self.weights = state.get("weights", {})
                self.history = state.get("history", [])
                self.pnl_history = state.get("pnl_history", {})
                self.pnl_net_history = state.get("pnl_net_history", {})
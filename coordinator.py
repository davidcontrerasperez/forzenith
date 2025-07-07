from collections import defaultdict
from agentes_zenith import AGENTES

class Coordinator:
    def __init__(self, tracker):
        self.tracker = tracker
        self.last_decisions = {}

    def decide(self, df, **kwargs):
        votes = defaultdict(float)
        decisiones = {}

        for nombre, agente in AGENTES.items():
            try:
                if nombre in ["multi_timeframe", "confluencia", "divergencia_tf"]:
                    decision = agente(kwargs["df_1h"], kwargs["df_4h"], kwargs.get("df_d", kwargs["df_4h"]))
                elif nombre == "mutante":
                    decision = agente(df, kwargs.get("iteracion", 0))
                elif nombre == "refuerzo":
                    decision = agente(df, kwargs.get("historial", []))
                elif nombre == "memoria":
                    decision = agente(df, kwargs.get("memoria", {}))
                elif nombre == "lunar":
                    decision = agente(kwargs.get("fase_actual", "nueva"))
                elif nombre == "evento":
                    decision = agente(kwargs.get("eventos", []))
                elif nombre in ["consenso", "discrepancia"]:
                    decision = agente(decisiones)
                elif nombre == "confianza":
                    decision = agente(self.tracker.weights)
                elif nombre == "nash":
                    decision = agente(decisiones, self.tracker.weights)
                else:
                    decision = agente(df)

                # Validación: si el agente devuelve un dict, extraer la clave 'decision'
                if isinstance(decision, dict):
                    print(f"⚠️ Agente '{nombre}' devolvió un dict: {decision}")
                    decision = decision.get("decision", "hold")

                # Validación extra: si no es string, forzar a 'hold'
                if not isinstance(decision, str):
                    print(f"⚠️ Agente '{nombre}' devolvió un tipo inválido: {type(decision)}")
                    decision = "hold"

            except Exception as e:
                print(f"❌ Error en agente '{nombre}': {e}")
                decision = "hold"

            decisiones[nombre] = decision
            peso = self.tracker.get_weight(nombre)
            votes[decision] += peso

        self.last_decisions = decisiones
        return max(votes, key=votes.get)

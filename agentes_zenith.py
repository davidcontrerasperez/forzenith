import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# ============================
# 📈 Indicadores técnicos
# ============================

def agente_rsi(df):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.iloc[-1] < 30:
        return "buy"
    elif rsi.iloc[-1] > 70:
        return "sell"
    return "hold"

def agente_macd(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-1] > signal.iloc[-1]:
        return "buy"
    elif macd.iloc[-1] < signal.iloc[-1]:
        return "sell"
    return "hold"

def agente_bollinger(df):
    ma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    price = df['close'].iloc[-1]
    if price < lower.iloc[-1]:
        return "buy"
    elif price > upper.iloc[-1]:
        return "sell"
    return "hold"

def agente_adx(df):
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(14).mean()
    if adx.iloc[-1] > 25:
        return "buy" if plus_di.iloc[-1] > minus_di.iloc[-1] else "sell"
    return "hold"

def agente_stochastic(df):
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    if k.iloc[-1] < 20:
        return "buy"
    elif k.iloc[-1] > 80:
        return "sell"
    return "hold"

def agente_ema_cruce(df):
    ema9 = df['close'].ewm(span=9).mean()
    ema21 = df['close'].ewm(span=21).mean()
    if ema9.iloc[-1] > ema21.iloc[-1]:
        return "buy"
    elif ema9.iloc[-1] < ema21.iloc[-1]:
        return "sell"
    return "hold"

def agente_candlestick(df):
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    ratio = body / range_
    if ratio.iloc[-1] < 0.3:
        if df['close'].iloc[-1] > df['open'].iloc[-1]:
            return "buy"
        else:
            return "sell"
    return "hold"

# ============================
# 📊 Agentes estadísticos
# ============================

def agente_zscore(df):
    mean = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    z = (df['close'] - mean) / std
    if z.iloc[-1] > 2:
        return "sell"
    elif z.iloc[-1] < -2:
        return "buy"
    return "hold"

def agente_mean_reversion(df):
    ma = df['close'].rolling(20).mean()
    price = df['close'].iloc[-1]
    if price < ma.iloc[-1] * 0.98:
        return "buy"
    elif price > ma.iloc[-1] * 1.02:
        return "sell"
    return "hold"

def agente_volatilidad(df):
    vol = df['close'].rolling(10).std().iloc[-1]
    if vol > 0.002:
        return "sell"
    elif vol < 0.001:
        return "buy"
    return "hold"

def agente_breakout(df):
    high = df['high'].rolling(20).max()
    low = df['low'].rolling(20).min()
    price = df['close'].iloc[-1]
    if price > high.iloc[-2]:
        return "buy"
    elif price < low.iloc[-2]:
        return "sell"
    return "hold"

def agente_range_bound(df):
    high = df['high'].rolling(20).max()
    low = df['low'].rolling(20).min()
    price = df['close'].iloc[-1]
    if price < low.iloc[-1] + (high.iloc[-1] - low.iloc[-1]) * 0.2:
        return "buy"
    elif price > high.iloc[-1] - (high.iloc[-1] - low.iloc[-1]) * 0.2:
        return "sell"
    return "hold"

# ============================
# 🤖 Agentes heurísticos
# ============================

def agente_random(df):
    return random.choice(["buy", "sell", "hold"])

def agente_persistente(df):
    if not hasattr(agente_persistente, "ultima"):
        agente_persistente.ultima = "hold"
    return agente_persistente.ultima

def agente_contrarian(df):
    return random.choice(["buy", "sell"])[::-1]

def agente_mayoria(df):
    return "hold"  # Placeholder

# ============================
# 🧬 Agentes tipo ML (simples)
# ============================

def agente_regresion_lineal(df):
    if len(df) < 20:
        return "hold"
    X = np.arange(20).reshape(-1, 1)
    y = df['close'].iloc[-20:].values
    model = LinearRegression().fit(X, y)
    pred = model.predict([[20]])
    if pred > y[-1]:
        return "buy"
    elif pred < y[-1]:
        return "sell"
    return "hold"
# ============================
# 🧠 Agentes adicionales
# ============================

def agente_gap(df):
    if len(df) < 2:
        return "hold"
    gap = df['open'].iloc[-1] - df['close'].iloc[-2]
    if gap < -0.001:
        return "buy"
    elif gap > 0.001:
        return "sell"
    return "hold"

def agente_inside_bar(df):
    if len(df) < 2:
        return "hold"
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if curr['high'] < prev['high'] and curr['low'] > prev['low']:
        return "buy"
    return "hold"

def agente_engulfing(df):
    if len(df) < 2:
        return "hold"
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if curr['close'] > curr['open'] and prev['close'] < prev['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
        return "buy"
    elif curr['close'] < curr['open'] and prev['close'] > prev['open'] and curr['close'] < prev['open'] and curr['open'] > prev['close']:
        return "sell"
    return "hold"

def agente_momentum_vela(df):
    body = abs(df['close'] - df['open'])
    rango = df['high'] - df['low']
    if body.iloc[-1] > 0.7 * rango.iloc[-1]:
        return "buy" if df['close'].iloc[-1] > df['open'].iloc[-1] else "sell"
    return "hold"

def agente_temporal(df):
    hora = pd.Timestamp.now().hour
    if 9 <= hora <= 11:
        return "buy"
    elif 15 <= hora <= 17:
        return "sell"
    return "hold"

def agente_volumen_climax(df):
    vol = df['volume']
    avg_vol = vol.rolling(20).mean()
    if vol.iloc[-1] > 2 * avg_vol.iloc[-1]:
        return "sell"
    elif vol.iloc[-1] < 0.5 * avg_vol.iloc[-1]:
        return "buy"
    return "hold"
def agente_consenso(decisiones):
    votos = list(decisiones.values())
    if votos.count("buy") > len(votos) * 0.6:
        return "buy"
    elif votos.count("sell") > len(votos) * 0.6:
        return "sell"
    return "hold"
def agente_multi_timeframe(df_1h, df_4h):
    signal_1h = agente_macd(df_1h)
    signal_4h = agente_macd(df_4h)
    if signal_1h == signal_4h:
        return signal_1h
    return "hold"
def agente_fibonacci(df):
    high = df['high'].rolling(50).max().iloc[-1]
    low = df['low'].rolling(50).min().iloc[-1]
    diff = high - low
    level_38 = high - 0.382 * diff
    level_61 = high - 0.618 * diff
    price = df['close'].iloc[-1]
    if price < level_61:
        return "buy"
    elif price > level_38:
        return "sell"
    return "hold"

# ============================
# 🧬 Agentes evolutivos / adaptativos
# ============================

def agente_mutante(df, iteracion):
    """
    Cambia de lógica cada 10 iteraciones entre momentum y reversión
    """
    if iteracion % 20 < 10:
        return "buy" if df['close'].iloc[-1] > df['close'].rolling(10).mean().iloc[-1] else "sell"
    else:
        return "buy" if df['close'].iloc[-1] < df['close'].rolling(10).mean().iloc[-1] else "sell"

def agente_refuerzo(df, historial=[]):
    """
    Refuerza decisiones pasadas si resultaron en PnL positivo
    """
    if historial and historial[-1]["pnl"] > 0:
        return historial[-1]["decision"]
    return random.choice(["buy", "sell", "hold"])

def agente_memoria(df, memoria={}):
    """
    Aprende de su propio historial de aciertos
    """
    if "aciertos" not in memoria:
        memoria["aciertos"] = {"buy": 0, "sell": 0, "hold": 0}
    mejor = max(memoria["aciertos"], key=memoria["aciertos"].get)
    return mejor

# ============================
# 🧠 Agentes de aprendizaje (simulados)
# ============================

def agente_svm_simulado(df):
    """
    Simula una predicción de SVM con una regla simple
    """
    cambio = df['close'].pct_change().iloc[-5:].mean()
    return "buy" if cambio > 0 else "sell"

def agente_random_forest_simulado(df):
    """
    Simula una predicción de Random Forest con múltiples condiciones
    """
    vol = df['close'].rolling(10).std().iloc[-1]
    tendencia = df['close'].iloc[-1] - df['close'].iloc[-10]
    if tendencia > 0 and vol < 0.01:
        return "buy"
    elif tendencia < 0 and vol > 0.01:
        return "sell"
    return "hold"

# ============================
# ⏳ Agentes multi-horizonte
# ============================

def agente_confluencia(df_1h, df_4h, df_d):
    """
    Solo actúa si todos los marcos temporales están de acuerdo
    """
    def tendencia(df):
        return "buy" if df['close'].iloc[-1] > df['close'].rolling(10).mean().iloc[-1] else "sell"
    señales = [tendencia(df_1h), tendencia(df_4h), tendencia(df_d)]
    if señales.count(señales[0]) == 3:
        return señales[0]
    return "hold"

def agente_divergencia_tf(df_1h, df_4h):
    """
    Opera si hay divergencia entre marcos temporales
    """
    def tendencia(df):
        return "buy" if df['close'].iloc[-1] > df['close'].rolling(10).mean().iloc[-1] else "sell"
    if tendencia(df_1h) != tendencia(df_4h):
        return "buy"
    return "hold"

# ============================
# 🧪 Agentes creativos / experimentales
# ============================

def agente_fibonacci(df):
    high = df['high'].rolling(50).max().iloc[-1]
    low = df['low'].rolling(50).min().iloc[-1]
    diff = high - low
    level_38 = high - 0.382 * diff
    level_61 = high - 0.618 * diff
    price = df['close'].iloc[-1]
    if price < level_61:
        return "buy"
    elif price > level_38:
        return "sell"
    return "hold"

def agente_fractal(df):
    """
    Detecta patrón fractal simple: pico o valle local
    """
    if len(df) < 5:
        return "hold"
    centro = df['close'].iloc[-3]
    if centro > df['close'].iloc[-4] and centro > df['close'].iloc[-2]:
        return "sell"
    elif centro < df['close'].iloc[-4] and centro < df['close'].iloc[-2]:
        return "buy"
    return "hold"

def agente_lunar(fase_actual):
    """
    fase_actual: string como 'nueva', 'llena', 'creciente', 'menguante'
    """
    if fase_actual == "llena":
        return "sell"
    elif fase_actual == "nueva":
        return "buy"
    return "hold"

# ============================
# 🧠 Agentes de control / metaconsenso
# ============================

def agente_consenso(decisiones):
    """
    decisiones: dict con decisiones de otros agentes
    """
    votos = list(decisiones.values())
    if votos.count("buy") > len(votos) * 0.6:
        return "buy"
    elif votos.count("sell") > len(votos) * 0.6:
        return "sell"
    return "hold"

def agente_discrepancia(decisiones):
    """
    Opera si hay mucha dispersión entre los agentes
    """
    votos = list(decisiones.values())
    if votos.count("buy") > 0 and votos.count("sell") > 0:
        return "sell"
    return "hold"

def agente_confianza(pesos):
    """
    pesos: dict con pesos de cada agente
    """
    max_peso = max(pesos.values())
    if max_peso > 0.5:
        return "buy"
    elif max_peso < 0.2:
        return "sell"
    return "hold"

import pandas as pd
import numpy as np
import random
from scipy.stats import norm

# ============================
# 🎯 Agente Nash (estrategia de equilibrio)
# ============================

def agente_nash(decisiones, pesos):
    """
    decisiones: dict con decisiones de otros agentes
    pesos: dict con pesos de cada agente
    """
    payoff = {"buy": 0, "sell": 0, "hold": 0}
    for agente, decision in decisiones.items():
        payoff[decision] += pesos.get(agente, 1.0)
    mejor = max(payoff, key=payoff.get)
    return mejor

# ============================
# 📊 Agente Bayesiano
# ============================

def agente_bayesiano(df):
    """
    Usa inferencia bayesiana simple para estimar probabilidad de subida
    """
    cambios = df['close'].pct_change().dropna()
    subida = cambios[cambios > 0]
    bajada = cambios[cambios < 0]

    p_subida = len(subida) / len(cambios)
    media = cambios.mean()
    std = cambios.std()

    prob = norm.cdf(0, loc=media, scale=std)
    if p_subida > 0.6 and prob < 0.4:
        return "buy"
    elif p_subida < 0.4 and prob > 0.6:
        return "sell"
    return "hold"

# ============================
# 📈 Agente VWAP (precio medio ponderado por volumen)
# ============================

def agente_vwap(df):
    """
    Opera si el precio actual está por encima o por debajo del VWAP
    """
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    precio = df['close'].iloc[-1]
    if precio > vwap.iloc[-1]:
        return "buy"
    elif precio < vwap.iloc[-1]:
        return "sell"
    return "hold"

# ============================
# 🧪 Agente de anomalías (detección de outliers)
# ============================

def agente_anomalia(df):
    """
    Detecta si el último cambio de precio es un outlier estadístico
    """
    cambios = df['close'].pct_change().dropna()
    z = (cambios - cambios.mean()) / cambios.std()
    if z.iloc[-1] > 2.5:
        return "sell"
    elif z.iloc[-1] < -2.5:
        return "buy"
    return "hold"

# ============================
# 📅 Agente de eventos programados
# ============================

def agente_evento(eventos):
    """
    eventos: lista de dicts con claves 'nombre', 'fecha', 'impacto'
    """
    hoy = pd.Timestamp.now().normalize()
    for evento in eventos:
        fecha = pd.to_datetime(evento["fecha"]).normalize()
        if fecha == hoy:
            if evento["impacto"] == "alto":
                return "sell"
            elif evento["impacto"] == "positivo":
                return "buy"
    return "hold"
# ============================
# 📦 Diccionario de agentes
# ============================

AGENTES = {
    # Técnicos
    "rsi": agente_rsi,
    "macd": agente_macd,
    "bollinger": agente_bollinger,
    "adx": agente_adx,
    "stochastic": agente_stochastic,
    "ema_cruce": agente_ema_cruce,
    "candlestick": agente_candlestick,

    # Estadísticos
    "zscore": agente_zscore,
    "mean_reversion": agente_mean_reversion,
    "volatilidad": agente_volatilidad,
    "breakout": agente_breakout,
    "range_bound": agente_range_bound,

    # Heurísticos
    "random": agente_random,
    "persistente": agente_persistente,
    "contrarian": agente_contrarian,
    "mayoria": agente_mayoria,

    # ML simple
    "regresion_lineal": agente_regresion_lineal,
    "svm_simulado": agente_svm_simulado,
    "random_forest_simulado": agente_random_forest_simulado,

    # Adicionales
    "gap": agente_gap,
    "inside_bar": agente_inside_bar,
    "engulfing": agente_engulfing,
    "momentum_vela": agente_momentum_vela,
    "temporal": agente_temporal,
    "volumen_climax": agente_volumen_climax,
    "multi_timeframe": agente_multi_timeframe,
    "fibonacci": agente_fibonacci,

    # Evolutivos
    "mutante": agente_mutante,
    "refuerzo": agente_refuerzo,
    "memoria": agente_memoria,

    # Multi-horizonte
    "confluencia": agente_confluencia,
    "divergencia_tf": agente_divergencia_tf,

    # Creativos
    "fractal": agente_fractal,
    "lunar": agente_lunar,

    # Metaconsenso
    "consenso": agente_consenso,
    "discrepancia": agente_discrepancia,
    "confianza": agente_confianza,

    # Avanzados
    "nash": agente_nash,
    "bayesiano": agente_bayesiano,
    "vwap": agente_vwap,
    "anomalia": agente_anomalia,
    "evento": agente_evento
}
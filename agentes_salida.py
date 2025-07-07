def salida_1(df):
    if len(df) < 2:
        return False
    return (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] > 0.01

def salida_2(df):
    if len(df) < 15:
        return False
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] > 70

def salida_3(df):
    if len(df) < 2:
        return False
    return (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] < -0.005

def salida_4(df):
    if len(df) < 3:
        return False
    ultimos = df.iloc[-3:]
    return (ultimos['close'] < ultimos['open']).all()

def salida_5(df):
    if len(df) < 10:
        return False
    ma = df['close'].rolling(10).mean()
    return df['close'].iloc[-1] < ma.iloc[-1]

# Diccionario de agentes de salida
agentes_salida = {
    "salida_1": salida_1,
    "salida_2": salida_2,
    "salida_3": salida_3,
    "salida_4": salida_4,
    "salida_5": salida_5
}
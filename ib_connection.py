import asyncio
import pandas as pd

# 🔧 Crear el bucle de eventos si no existe (para compatibilidad con Streamlit)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

ib = None  # Se inicializa dinámicamente

def connect_ib(mode="demo", client_id=1):
    global ib
    from ib_insync import IB
    ib = IB()
    port = 7497 if mode == "demo" else 7496
    try:
        ib.connect("127.0.0.1", port, clientId=client_id)
        print(f"✅ Conectado a IBKR en modo {mode.upper()} con clientId={client_id}")
        return True
    except Exception as e:
        print(f"❌ Error al conectar con IBKR: {e}")
        return False

def disconnect_ib():
    global ib
    if ib and ib.isConnected():
        ib.disconnect()
        print("🔌 Desconectado de IBKR")

def get_latest_data(symbol="EURUSD"):
    global ib
    from ib_insync import Forex, util  # Importación diferida
    if not ib or not ib.isConnected():
        raise RuntimeError("IBKR no está conectado. Llama a connect_ib() primero.")

    contract = Forex(symbol)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='2 D',
        barSizeSetting='1 min',
        whatToShow='MIDPOINT',
        useRTH=False,
        formatDate=1
    )
    df = util.df(bars)
    df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}, inplace=True)
    df["volume"] = 1000  # IBKR no proporciona volumen en MIDPOINT
    return df.tail(100)
def get_positions():
    global ib
    from ib_insync import util
    if not ib or not ib.isConnected():
        raise RuntimeError("IBKR no está conectado.")

    positions = ib.positions()
    if not positions:
        return pd.DataFrame(columns=["symbol", "position", "avgPrice"])

    data = []
    for pos in positions:
        contract = pos.contract
        symbol = f"{contract.symbol}.{contract.currency}"
        position = pos.position
        avg_price = getattr(pos, "avgPrice", None)
        data.append({
            "symbol": symbol,
            "position": position,
            "avgPrice": round(avg_price, 5) if avg_price else "N/A"
        })

    return pd.DataFrame(data)

def get_account_summary():
    global ib
    from ib_insync import util
    if not ib or not ib.isConnected():
        raise RuntimeError("IBKR no está conectado.")

    summary_list = ib.accountSummary()
    df = util.df(summary_list)

    if "tag" not in df.columns or "value" not in df.columns:
        raise ValueError("No se pudo interpretar el resumen de cuenta.")

    net_liq_row = df[df["tag"] == "NetLiquidation"]
    if net_liq_row.empty:
        raise ValueError("No se encontró el valor NetLiquidation.")

    net_liq = float(net_liq_row["value"].values[0])
    currency = net_liq_row["currency"].values[0]
    account_id = net_liq_row["account"].values[0]
    return net_liq, currency, account_id
def execute_order(symbol="EURUSD", action="BUY", quantity=100000):
    global ib
    from ib_insync import Forex, MarketOrder

    if not ib or not ib.isConnected():
        raise RuntimeError("IBKR no está conectado.")

    contract = Forex(symbol)
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)  # Esperar a que se procese
    status = trade.orderStatus.status
    print(f"🟢 Orden enviada: {action} {quantity} {symbol} — Estado: {status}")
    return status
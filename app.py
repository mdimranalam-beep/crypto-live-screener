import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Crypto Live Screener",
    page_icon="₿",
    layout="wide"
)

# -----------------------------
# SETTINGS
# -----------------------------

BINANCE = "https://api.binance.com"

TOP10 = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "AVAXUSDT",
    "LINKUSDT"
]

TOP20_EXTRA = [
    "DOTUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
    "NEARUSDT"
]


# -----------------------------
# BINANCE DATA
# -----------------------------

@st.cache_data(ttl=5)
def get_price_data():
    url = BINANCE + "/api/v3/ticker/24hr"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=15)
def get_klines(symbol, interval, limit=200):
    url = BINANCE + "/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "buy_volume",
        "buy_quote_volume",
        "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    return df


# -----------------------------
# INDICATORS
# -----------------------------

def add_indicators(df):

    df = df.copy()

    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # Supertrend
    multiplier = 3

    hl2 = (df["high"] + df["low"]) / 2

    upper = hl2 + multiplier * df["ATR"]
    lower = hl2 - multiplier * df["ATR"]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    direction.iloc[0] = 1
    supertrend.iloc[0] = lower.iloc[0]

    for i in range(1, len(df)):

        if df["close"].iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1

        elif df["close"].iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1

        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower.iloc[i]
        else:
            supertrend.iloc[i] = upper.iloc[i]

    df["Supertrend"] = supertrend
    df["ST_Direction"] = direction

    # Volume average
    df["VolumeMA"] = df["volume"].rolling(20).mean()

    # EMA slope
    df["EMA20_Slope"] = df["EMA20"].pct_change(5) * 100

    return df


# -----------------------------
# TIMEFRAME BIAS
# -----------------------------

def timeframe_bias(df):

    last = df.iloc[-1]

    score = 0

    if last["close"] > last["EMA20"]:
        score += 1
    else:
        score -= 1

    if last["EMA20"] > last["EMA50"]:
        score += 1
    else:
        score -= 1

    if last["EMA50"] > last["EMA200"]:
        score += 1
    else:
        score -= 1

    if last["EMA20_Slope"] > 0:
        score += 1
    else:
        score -= 1

    if score >= 3:
        return "BULLISH", score

    if score <= -3:
        return "BEARISH", score

    return "NEUTRAL", score


# -----------------------------
# SIGNAL ENGINE
# -----------------------------

def calculate_signal(df_5m, df_1h, df_4h):

    a = df_5m.iloc[-1]

    bias_1h, score_1h = timeframe_bias(df_1h)
    bias_4h, score_4h = timeframe_bias(df_4h)

    score = 0

    # 1H bias
    if bias_1h == "BULLISH":
        score += 2
    elif bias_1h == "BEARISH":
        score -= 2

    # 4H bias
    if bias_4h == "BULLISH":
        score += 2
    elif bias_4h == "BEARISH":
        score -= 2

    # EMA alignment
    if a["EMA20"] > a["EMA50"] > a["EMA200"]:
        score += 2

    elif a["EMA20"] < a["EMA50"] < a["EMA200"]:
        score -= 2

    # Supertrend
    if a["ST_Direction"] == 1:
        score += 2
    else:
        score -= 2

    # RSI
    if 50 < a["RSI"] < 70:
        score += 1

    elif 30 < a["RSI"] < 50:
        score -= 1

    # Volume confirmation
    if a["volume"] > a["VolumeMA"]:
        if a["close"] > a["open"]:
            score += 1
        else:
            score -= 1

    if score >= 6:
        signal = "BUY"

    elif score <= -6:
        signal = "SELL"

    else:
        signal = "WAIT"

    return signal, score, bias_1h, bias_4h


# -----------------------------
# MAIN DASHBOARD
# -----------------------------

st.title("₿ Crypto Live Trading Screener")

st.caption(
    "Multi-timeframe crypto scanner • 4H + 1H bias • 5M entry • "
    "EMA + RSI + Supertrend + Volume"
)

# Controls

col1, col2, col3 = st.columns(3)

with col1:
    universe = st.selectbox(
        "Crypto universe",
        ["Top 10", "Top 20"]
    )

with col2:
    refresh = st.selectbox(
        "Refresh",
        ["5 seconds", "10 seconds", "15 seconds", "30 seconds"]
    )

with col3:
    min_score = st.slider(
        "Minimum signal score",
        0,
        10,
        6
    )

if universe == "Top 10":
    symbols = TOP10
else:
    symbols = TOP10 + TOP20_EXTRA


# -----------------------------
# MARKET BIAS
# -----------------------------

try:

    btc_4h = add_indicators(
        get_klines("BTCUSDT", "4h")
    )

    btc_1h = add_indicators(
        get_klines("BTCUSDT", "1h")
    )

    btc_bias_4h, _ = timeframe_bias(btc_4h)
    btc_bias_1h, _ = timeframe_bias(btc_1h)

    if btc_bias_4h == "BULLISH" and btc_bias_1h == "BULLISH":
        market_bias = "🟢 BULLISH"

    elif btc_bias_4h == "BEARISH" and btc_bias_1h == "BEARISH":
        market_bias = "🔴 BEARISH"

    else:
        market_bias = "🟡 MIXED"

except Exception:
    market_bias = "⚪ UNAVAILABLE"


st.subheader("Overall Crypto Market")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Market Bias", market_bias)

with m2:
    st.metric("Primary TF", "4H")

with m3:
    st.metric("Entry TF", "5M")


# -----------------------------
# SCANNER
# -----------------------------

st.subheader("Live Scanner")

rows = []

try:

    prices = get_price_data()

    price_map = {
        x["symbol"]: x
        for x in prices
        if x["symbol"] in symbols
    }

    for symbol in symbols:

        try:

            df5 = add_indicators(
                get_klines(symbol, "5m")
            )

            df1 = add_indicators(
                get_klines(symbol, "1h")
            )

            df4 = add_indicators(
                get_klines(symbol, "4h")
            )

            signal, score, bias1, bias4 = calculate_signal(
                df5,
                df1,
                df4
            )

            last = df5.iloc[-1]

            ticker = price_map.get(symbol, {})

            price = float(ticker.get(
                "lastPrice",
                last["close"]
            ))

            change = float(ticker.get(
                "priceChangePercent",
                0
            ))

            if signal == "BUY":
                action = "🟢 BUY"

            elif signal == "SELL":
                action = "🔴 SELL"

            else:
                action = "⚪ WAIT"

            if abs(score) >= min_score or signal == "WAIT":

                rows.append({
                    "Crypto": symbol.replace("USDT", ""),
                    "Price": round(price, 4),
                    "24H %": round(change, 2),
                    "4H Bias": bias4,
                    "1H Bias": bias1,
                    "5M RSI": round(float(last["RSI"]), 1),
                    "EMA Slope": round(float(last["EMA20_Slope"]), 2),
                    "Signal": action,
                    "Score": score
                })

        except Exception:
            continue

    result = pd.DataFrame(rows)

    if not result.empty:

        result["Rank"] = result["Score"].abs()

        result = result.sort_values(
            "Rank",
            ascending=False
        )

        result = result.drop(
            columns=["Rank"]
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Higher absolute score = stronger indicator confluence. "
            "Signals are informational, not financial advice."
        )

    else:
        st.warning(
            "No market data available right now. Please wait a few seconds."
        )

except Exception as e:

    st.error(
        "Unable to load market data. Please try again."
    )


# -----------------------------
# TRADINGVIEW LINKS
# -----------------------------

st.subheader("Charts")

chart_cols = st.columns(5)

for i, symbol in enumerate(symbols[:10]):

    with chart_cols[i % 5]:

        url = (
            "https://www.tradingview.com/chart/"
            "?symbol=BINANCE:"
            + symbol
        )

        st.link_button(
            symbol.replace("USDT", ""),
            url
        )


st.divider()

st.caption(
    "Data source: Binance public market data. "
    "This dashboard does not place trades."
)

# Auto refresh
try:
    seconds = int(refresh.split()[0])

    @st.fragment(run_every=f"{seconds}s")
    def live_refresh():
        st.empty()

    live_refresh()

except Exception:
    pass

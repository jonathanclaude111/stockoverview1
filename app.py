"""
AI-Aandelen Dashboard
----------------------
Streamlit-app die fundamentele data ophaalt via yfinance (Yahoo Finance),
zonder dat er een API-key nodig is.

Lokaal draaien:
    pip install -r requirements.txt
    streamlit run app.py

Hosten: push deze map naar een GitHub-repository en koppel die aan
Streamlit Community Cloud (share.streamlit.io).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="AI-Aandelen Dashboard", layout="wide")

TICKERS = [
    "NVDA", "MU", "GOOGL", "TSLA", "MSFT", "AVGO", "META", "ARM", "QCOM", "AMD",
    "WDC", "AXTI", "APH", "ATEN", "ALAB", "APLD", "CIEN", "COHR", "CRDO", "DELL",
    "VRT", "NBIS", "PDD", "AAPL", "SOF.BR", "BABA", "LLY", "TSM", "PLTR", "CBRS",
    "CRWV", "QBTS", "MVIS", "ADUR", "POET", "IBM", "SPCX", "AMZN", "ORCL", "MRVL",
    "SNDK", "ANET", "SMCI", "AMAT", "LRCX", "KLAC", "INTC",
]

CACHE_TTL = 15 * 60  # 15 minuten


def fmt_num(x, prefix="", suffix="", decimals=2):
    """Formatteer een getal netjes, of geef 'n.b.' terug als het ontbreekt."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n.b."
    try:
        if abs(x) >= 1e12:
            return f"{prefix}{x/1e12:.{decimals}f}T{suffix}"
        if abs(x) >= 1e9:
            return f"{prefix}{x/1e9:.{decimals}f}B{suffix}"
        if abs(x) >= 1e6:
            return f"{prefix}{x/1e6:.{decimals}f}M{suffix}"
        return f"{prefix}{x:,.{decimals}f}{suffix}"
    except Exception:
        return "n.b."


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_stock_info(ticker: str) -> dict:
    """Haalt alle relevante velden op voor één ticker. Faalt stil per veld."""
    data = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}

    data["name"] = info.get("longName") or info.get("shortName") or ticker
    data["sector"] = info.get("sector", "n.b.")
    data["industry"] = info.get("industry", "n.b.")
    data["summary"] = info.get("longBusinessSummary", "Geen beschrijving beschikbaar.")

    data["price"] = info.get("currentPrice") or info.get("regularMarketPrice")
    data["currency"] = info.get("currency", "")
    data["market_cap"] = info.get("marketCap")

    data["pe_trailing"] = info.get("trailingPE")
    data["pe_forward"] = info.get("forwardPE")
    data["peg_ratio"] = info.get("pegRatio") or info.get("trailingPegRatio")

    data["52w_low"] = info.get("fiftyTwoWeekLow")
    data["52w_high"] = info.get("fiftyTwoWeekHigh")

    data["target_mean"] = info.get("targetMeanPrice")
    data["target_low"] = info.get("targetLowPrice")
    data["target_high"] = info.get("targetHighPrice")
    data["recommendation"] = info.get("recommendationKey", "n.b.")
    data["num_analysts"] = info.get("numberOfAnalystOpinions")

    # Free Cash Flow (TTM) — via cashflow-statement
    fcf = None
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            if "Free Cash Flow" in cf.index:
                fcf = cf.loc["Free Cash Flow"].iloc[0]
            elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
                fcf = cf.loc["Operating Cash Flow"].iloc[0] + cf.loc["Capital Expenditure"].iloc[0]
    except Exception:
        pass
    data["fcf_ttm"] = fcf

    # Insider transacties (laatste paar)
    insiders = None
    try:
        ins = t.insider_transactions
        if ins is not None and not ins.empty:
            insiders = ins.head(5)
    except Exception:
        pass
    data["insider_transactions"] = insiders

    # YTD-rendement
    ytd_return = None
    try:
        hist = t.history(period="ytd")
        if not hist.empty:
            start = hist["Close"].iloc[0]
            end = hist["Close"].iloc[-1]
            ytd_return = (end / start - 1) * 100
    except Exception:
        pass
    data["ytd_return"] = ytd_return

    return data


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_all_stocks(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        d = get_stock_info(tk)
        rows.append(d)
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_correlation_matrix(tickers: list[str]) -> pd.DataFrame:
    try:
        raw = yf.download(tickers, period="6mo", interval="1d", progress=False)["Close"]
        returns = raw.pct_change().dropna(how="all")
        corr = returns.corr()
        return corr
    except Exception:
        return pd.DataFrame()


def rating_badge(rec: str) -> str:
    mapping = {
        "strong_buy": "🟢 Strong Buy",
        "buy": "🟢 Buy",
        "hold": "🟡 Hold",
        "underperform": "🔴 Underperform",
        "sell": "🔴 Sell",
        "none": "n.b.",
    }
    return mapping.get((rec or "").lower(), rec or "n.b.")


# ----------------------- UI -----------------------

st.title("📊 AI-Aandelen Dashboard")
st.caption(
    f"Data via Yahoo Finance (yfinance) — laatst opgehaald rond "
    f"{datetime.now().strftime('%d/%m/%Y %H:%M')}. Koersen kunnen ~15 min vertraagd zijn."
)

tab_overview, tab_detail, tab_corr = st.tabs(["📋 Overzicht", "🔍 Detail per aandeel", "🔗 Correlatie"])

with st.spinner("Data ophalen voor alle aandelen... (kan even duren bij eerste keer)"):
    df = get_all_stocks(TICKERS)

with tab_overview:
    st.subheader("Overzichtstabel")
    display_df = df.copy()
    display_df["Koers"] = display_df.apply(lambda r: fmt_num(r["price"], suffix=f" {r['currency']}"), axis=1)
    display_df["Marktkap"] = display_df["market_cap"].apply(fmt_num)
    display_df["P/E (trailing)"] = display_df["pe_trailing"].apply(lambda x: fmt_num(x))
    display_df["P/E (forward)"] = display_df["pe_forward"].apply(lambda x: fmt_num(x))
    display_df["PEG"] = display_df["peg_ratio"].apply(lambda x: fmt_num(x))
    display_df["FCF (TTM)"] = display_df["fcf_ttm"].apply(fmt_num)
    display_df["YTD %"] = display_df["ytd_return"].apply(lambda x: fmt_num(x, suffix="%"))
    display_df["Koersdoel (gem.)"] = display_df["target_mean"].apply(fmt_num)
    display_df["Rating"] = display_df["recommendation"].apply(rating_badge)

    cols_to_show = [
        "ticker", "name", "Koers", "Marktkap", "P/E (trailing)", "P/E (forward)",
        "PEG", "FCF (TTM)", "YTD %", "Koersdoel (gem.)", "Rating",
    ]
    st.dataframe(
        display_df[cols_to_show].rename(columns={"ticker": "Ticker", "name": "Naam"}),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

with tab_detail:
    st.subheader("Detail per aandeel")
    ticker_choice = st.selectbox("Kies een aandeel", TICKERS)
    d = get_stock_info(ticker_choice)

    col1, col2, col3 = st.columns(3)
    col1.metric("Koers", fmt_num(d["price"], suffix=f" {d['currency']}"))
    col2.metric("Marktkap", fmt_num(d["market_cap"]))
    col3.metric("YTD-rendement", fmt_num(d["ytd_return"], suffix="%"))

    st.markdown(f"### {d['name']} ({d['ticker']})")
    st.markdown(f"**Sector:** {d['sector']} — **Industrie:** {d['industry']}")
    st.markdown("**Activiteit:**")
    st.write(d["summary"])

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P/E trailing", fmt_num(d["pe_trailing"]))
    c2.metric("P/E forward", fmt_num(d["pe_forward"]))
    c3.metric("PEG-ratio", fmt_num(d["peg_ratio"]))
    c4.metric("FCF (TTM)", fmt_num(d["fcf_ttm"]))

    c5, c6 = st.columns(2)
    c5.metric("52-weekbereik", f"{fmt_num(d['52w_low'])} — {fmt_num(d['52w_high'])}")
    c6.metric("Rating analisten", rating_badge(d["recommendation"]))

    st.markdown("**Koersdoelen analisten:**")
    st.write(
        f"Gemiddeld: {fmt_num(d['target_mean'])} — "
        f"Laag: {fmt_num(d['target_low'])} — "
        f"Hoog: {fmt_num(d['target_high'])} "
        f"(gebaseerd op {d['num_analysts'] if d['num_analysts'] else 'n.b.'} analisten)"
    )

    st.markdown("**Recente insider-transacties:**")
    if d["insider_transactions"] is not None and not d["insider_transactions"].empty:
        st.dataframe(d["insider_transactions"], use_container_width=True, hide_index=True)
    else:
        st.write("Geen data beschikbaar.")

with tab_corr:
    st.subheader("Onderlinge correlatie (obv 6 maanden dagelijkse rendementen)")
    with st.spinner("Correlatiematrix berekenen..."):
        corr = get_correlation_matrix(TICKERS)
    if not corr.empty:
        st.dataframe(
            corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True,
            height=700,
        )
    else:
        st.write("Kon correlatiematrix niet berekenen.")

st.markdown("---")
st.caption(
    "⚠️ Deze data is louter informatief en vormt geen beleggingsadvies. "
    "Sommige tickers uit de lijst worden mogelijk niet (correct) herkend door Yahoo Finance."
)

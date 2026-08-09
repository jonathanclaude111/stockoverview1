"""
AI Stock Dashboard
--------------------
Streamlit app that pulls fundamental data via yfinance (Yahoo Finance),
no API key required for the market data itself.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Hosting: push this folder to a GitHub repository and connect it to
Streamlit Community Cloud (share.streamlit.io).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import anthropic
import json
import os

st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

CLAUDE_MODEL = "claude-sonnet-5"
TICKERS_FILE = os.path.join(os.path.dirname(__file__), "tickers.json")


def get_claude_client():
    """Returns an Anthropic client, or None if no API key is configured."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def extract_text(response) -> str:
    """Extracts the text from a Claude response, regardless of any 'thinking' blocks before it."""
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if parts:
        return "\n".join(parts)
    block_types = [getattr(b, "type", "unknown") for b in response.content]
    return (
        "⚠️ No text response received from Claude. "
        f"(stop_reason: {getattr(response, 'stop_reason', 'n/a')}, blocks: {block_types}). "
        "max_tokens may have been too low — try again."
    )


DEFAULT_TICKERS = [
    "NVDA", "MU", "GOOGL", "TSLA", "MSFT", "AVGO", "META", "ARM", "QCOM", "AMD",
    "WDC", "AXTI", "APH", "ATEN", "ALAB", "APLD", "CIEN", "COHR", "CRDO", "DELL",
    "VRT", "NBIS", "PDD", "AAPL", "SOF.BR", "BABA", "LLY", "TSM", "PLTR", "CBRS",
    "CRWV", "QBTS", "MVIS", "ADUR", "POET", "IBM", "SPCX", "AMZN", "ORCL", "MRVL",
    "SNDK", "ANET", "SMCI", "AMAT", "LRCX", "KLAC", "INTC",
]


def load_tickers() -> list[str]:
    """Reads the ticker list from disk, or returns the default list if no file exists yet."""
    try:
        if os.path.exists(TICKERS_FILE):
            with open(TICKERS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data
    except Exception:
        pass
    return DEFAULT_TICKERS.copy()


def save_tickers(tickers: list[str]) -> None:
    """Writes the ticker list to disk so it persists across reloads."""
    try:
        with open(TICKERS_FILE, "w") as f:
            json.dump(tickers, f)
    except Exception as e:
        st.warning(f"Could not save ticker list: {e}")


CACHE_TTL = 15 * 60  # 15 minutes


def fmt_num(x, prefix="", suffix="", decimals=2):
    """Formats a number nicely, or returns 'n/a' if it's missing."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    try:
        if abs(x) >= 1e12:
            return f"{prefix}{x/1e12:.{decimals}f}T{suffix}"
        if abs(x) >= 1e9:
            return f"{prefix}{x/1e9:.{decimals}f}B{suffix}"
        if abs(x) >= 1e6:
            return f"{prefix}{x/1e6:.{decimals}f}M{suffix}"
        return f"{prefix}{x:,.{decimals}f}{suffix}"
    except Exception:
        return "n/a"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_stock_info(ticker: str) -> dict:
    """Fetches all relevant fields for one ticker. Fails silently per-field."""
    data = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}

    data["name"] = info.get("longName") or info.get("shortName") or ticker
    data["sector"] = info.get("sector", "n/a")
    data["industry"] = info.get("industry", "n/a")
    data["summary"] = info.get("longBusinessSummary", "No description available.")

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
    data["recommendation"] = info.get("recommendationKey", "n/a")
    data["num_analysts"] = info.get("numberOfAnalystOpinions")

    # Free Cash Flow (TTM) — from the cash flow statement
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

    # Insider transactions (most recent few)
    insiders = None
    try:
        ins = t.insider_transactions
        if ins is not None and not ins.empty:
            insiders = ins.head(5)
    except Exception:
        pass
    data["insider_transactions"] = insiders

    # YTD return
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_claude_analysis(ticker: str, snapshot: dict) -> str:
    """Asks Claude to interpret the numbers and give a buy-worthiness assessment.
    snapshot contains only plain values (no DataFrames) so it's cacheable/serializable."""
    client = get_claude_client()
    if client is None:
        return "⚠️ No ANTHROPIC_API_KEY configured — see instructions in the README."

    prompt = f"""You are a level-headed, critical stock analyst. Below are the current
fundamental figures for {snapshot.get('name')} ({ticker}). Give a short, concrete analysis in English:

1. Interpretation of the figures (valuation, growth, health) — 3-4 sentences
2. Arguments suggesting the stock is undervalued
3. Arguments suggesting the stock is overvalued
4. A cautious estimate of an attractive entry level (price or multiple), with a clear note
   that this is not financial advice, just an educational estimate based on the figures

Figures:
- Price: {snapshot.get('price')} {snapshot.get('currency')}
- Market cap: {snapshot.get('market_cap')}
- P/E trailing: {snapshot.get('pe_trailing')} — P/E forward: {snapshot.get('pe_forward')}
- PEG ratio: {snapshot.get('peg_ratio')}
- FCF (TTM): {snapshot.get('fcf_ttm')}
- 52-week range: {snapshot.get('52w_low')} - {snapshot.get('52w_high')}
- YTD return: {snapshot.get('ytd_return')}%
- Analyst target price avg/low/high: {snapshot.get('target_mean')} / {snapshot.get('target_low')} / {snapshot.get('target_high')}
- Analyst rating: {snapshot.get('recommendation')} (based on {snapshot.get('num_analysts')} analysts)
- Sector: {snapshot.get('sector')} — Industry: {snapshot.get('industry')}

Keep it compact (max ~250 words), use short paragraphs with bold subheadings."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text(response)
    except Exception as e:
        return f"⚠️ Error retrieving AI analysis: {e}"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_top_picks(strong_buy_rows: list[dict]) -> str:
    """Asks Claude to pick the 3 most interesting stocks among the Strong Buy-rated ones."""
    client = get_claude_client()
    if client is None:
        return "⚠️ No ANTHROPIC_API_KEY configured — see instructions in the README."

    lines = []
    for r in strong_buy_rows:
        upside = None
        if r.get("price") and r.get("target_mean"):
            try:
                upside = (r["target_mean"] / r["price"] - 1) * 100
            except Exception:
                upside = None
        lines.append(
            f"- {r['ticker']} ({r['name']}): price {r.get('price')}, "
            f"P/E trailing {r.get('pe_trailing')}, P/E forward {r.get('pe_forward')}, "
            f"PEG {r.get('peg_ratio')}, FCF TTM {r.get('fcf_ttm')}, YTD {r.get('ytd_return')}%, "
            f"avg target price {r.get('target_mean')} (upside ~{upside:.0f}% if known)"
        )
    stock_list_text = "\n".join(lines)

    prompt = f"""You are a critical, level-headed stock analyst. Below are ALL stocks from my
watchlist that currently have a 'Strong Buy' rating from analysts. Pick the 3 most interesting
ones to buy RIGHT NOW, and explain why.

Important:
- Be critical: ignore or explicitly flag stocks with negative earnings/FCF, extreme P/E's,
  or unrealistic price targets as (too) speculative
- Favor a good balance between valuation (low PEG/forward P/E), strong FCF, and upside potential
- Explicitly state that this is an educational, numbers-based interpretation, not financial advice
- Answer in English, compact, with a short rationale (2-3 sentences) for each of the top 3 picks

Stocks:
{stock_list_text}"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text(response)
    except Exception as e:
        return f"⚠️ Error retrieving AI analysis: {e}"


def rating_badge(rec: str) -> str:
    mapping = {
        "strong_buy": "🟢 Strong Buy",
        "buy": "🟢 Buy",
        "hold": "🟡 Hold",
        "underperform": "🔴 Underperform",
        "sell": "🔴 Sell",
        "none": "n/a",
    }
    return mapping.get((rec or "").lower(), rec or "n/a")


# ----------------------- Ticker management (sidebar) -----------------------

if "tickers" not in st.session_state:
    st.session_state.tickers = load_tickers()

with st.sidebar:
    st.header("⚙️ Manage stock list")

    new_ticker = st.text_input("Add ticker (e.g. NFLX)", key="new_ticker_input").strip().upper()
    if st.button("➕ Add"):
        if not new_ticker:
            st.warning("Please enter a ticker first.")
        elif new_ticker in st.session_state.tickers:
            st.warning(f"{new_ticker} is already in the list.")
        else:
            st.session_state.tickers.append(new_ticker)
            save_tickers(st.session_state.tickers)
            st.success(f"{new_ticker} added and saved.")
            st.rerun()

    st.markdown("---")
    to_remove = st.multiselect("Remove tickers", options=st.session_state.tickers)
    if st.button("🗑️ Remove selected") and to_remove:
        st.session_state.tickers = [t for t in st.session_state.tickers if t not in to_remove]
        save_tickers(st.session_state.tickers)
        st.success(f"Removed and saved: {', '.join(to_remove)}")
        st.rerun()

    st.markdown("---")
    if st.button("↩️ Reset to default list"):
        st.session_state.tickers = DEFAULT_TICKERS.copy()
        save_tickers(st.session_state.tickers)
        st.rerun()

    st.caption(
        f"Current list: {len(st.session_state.tickers)} stocks. Changes persist "
        "after reloading, but may reset after the app restarts on Streamlit Cloud. "
        "⚠️ New tickers are not validated upfront — if the ticker doesn't exist on "
        "Yahoo Finance, you'll see 'n/a' in the table instead of an error."
    )

TICKERS = st.session_state.tickers

# ----------------------- UI -----------------------

st.title("📊 AI Stock Dashboard")
st.caption(
    f"Data via Yahoo Finance (yfinance) — last fetched around "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}. Prices may be delayed ~15 min."
)

tab_overview, tab_detail, tab_corr = st.tabs(["📋 Overview", "🔍 Stock detail", "🔗 Correlation"])

with st.spinner("Fetching data for all stocks... (may take a moment on first load)"):
    df = get_all_stocks(TICKERS)

with tab_overview:
    st.subheader("Overview table")
    display_df = df.copy()
    display_df["Price"] = display_df.apply(lambda r: fmt_num(r["price"], suffix=f" {r['currency']}"), axis=1)
    display_df["Market Cap"] = display_df["market_cap"].apply(fmt_num)
    display_df["P/E (trailing)"] = display_df["pe_trailing"].apply(lambda x: fmt_num(x))
    display_df["P/E (forward)"] = display_df["pe_forward"].apply(lambda x: fmt_num(x))
    display_df["PEG"] = display_df["peg_ratio"].apply(lambda x: fmt_num(x))
    display_df["FCF (TTM)"] = display_df["fcf_ttm"].apply(fmt_num)
    display_df["YTD %"] = display_df["ytd_return"].apply(lambda x: fmt_num(x, suffix="%"))
    display_df["Avg. Target"] = display_df["target_mean"].apply(fmt_num)
    display_df["Rating"] = display_df["recommendation"].apply(rating_badge)

    cols_to_show = [
        "ticker", "name", "Price", "Market Cap", "P/E (trailing)", "P/E (forward)",
        "PEG", "FCF (TTM)", "YTD %", "Avg. Target", "Rating",
    ]
    st.dataframe(
        display_df[cols_to_show].rename(columns={"ticker": "Ticker", "name": "Name"}),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.markdown("---")
    st.markdown("### 🏆 Top 3 Strong Buy candidates")
    st.caption(
        "Filters all stocks with a 'Strong Buy' rating and asks Claude to select the 3 most "
        "interesting ones based on valuation, FCF, and upside. Not financial advice."
    )
    if st.button("Generate top 3"):
        strong_buy_df = df[df["recommendation"].str.lower() == "strong_buy"]
        if strong_buy_df.empty:
            st.write("No stocks with a 'Strong Buy' rating found in the current data.")
        else:
            strong_buy_rows = strong_buy_df.drop(columns=["insider_transactions"], errors="ignore").to_dict("records")
            with st.spinner(f"Claude is analyzing {len(strong_buy_rows)} Strong Buy stocks..."):
                top_picks = get_top_picks(strong_buy_rows)
            st.session_state["top_picks_text"] = top_picks

    if "top_picks_text" in st.session_state:
        st.markdown(st.session_state["top_picks_text"])

with tab_detail:
    st.subheader("Stock detail")
    ticker_choice = st.selectbox("Choose a stock", TICKERS)
    d = get_stock_info(ticker_choice)

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", fmt_num(d["price"], suffix=f" {d['currency']}"))
    col2.metric("Market Cap", fmt_num(d["market_cap"]))
    col3.metric("YTD return", fmt_num(d["ytd_return"], suffix="%"))

    st.markdown(f"### {d['name']} ({d['ticker']})")
    st.markdown(f"**Sector:** {d['sector']} — **Industry:** {d['industry']}")
    st.markdown("**Business description:**")
    st.write(d["summary"])

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P/E trailing", fmt_num(d["pe_trailing"]))
    c2.metric("P/E forward", fmt_num(d["pe_forward"]))
    c3.metric("PEG ratio", fmt_num(d["peg_ratio"]))
    c4.metric("FCF (TTM)", fmt_num(d["fcf_ttm"]))

    c5, c6 = st.columns(2)
    c5.metric("52-week range", f"{fmt_num(d['52w_low'])} — {fmt_num(d['52w_high'])}")
    c6.metric("Analyst rating", rating_badge(d["recommendation"]))

    st.markdown("**Analyst price targets:**")
    st.write(
        f"Average: {fmt_num(d['target_mean'])} — "
        f"Low: {fmt_num(d['target_low'])} — "
        f"High: {fmt_num(d['target_high'])} "
        f"(based on {d['num_analysts'] if d['num_analysts'] else 'n/a'} analysts)"
    )

    st.markdown("**Recent insider transactions:**")
    if d["insider_transactions"] is not None and not d["insider_transactions"].empty:
        st.dataframe(d["insider_transactions"], use_container_width=True, hide_index=True)
    else:
        st.write("No data available.")

    st.markdown("---")
    st.markdown("### 🤖 AI interpretation & buy-worthiness")
    st.caption(
        "Generated by Claude based on the figures above. "
        "This is not financial advice, just an educational estimate."
    )
    if st.button("Generate AI analysis", key=f"analyse_{ticker_choice}"):
        # Only pass plain, serializable values to the cache function
        snapshot = {k: v for k, v in d.items() if k != "insider_transactions"}
        with st.spinner("Claude is analyzing the figures..."):
            analysis = get_claude_analysis(ticker_choice, snapshot)
        st.session_state[f"analysis_text_{ticker_choice}"] = analysis

    analysis_key = f"analysis_text_{ticker_choice}"
    if analysis_key in st.session_state:
        st.markdown(st.session_state[analysis_key])

with tab_corr:
    st.subheader("Correlation (based on 6 months of daily returns)")
    with st.spinner("Calculating correlation matrix..."):
        corr = get_correlation_matrix(TICKERS)
    if not corr.empty:
        st.dataframe(
            corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True,
            height=700,
        )
    else:
        st.write("Could not calculate the correlation matrix.")

st.markdown("---")
st.caption(
    "⚠️ This data is for informational purposes only and does not constitute investment advice. "
    "Some tickers in the list may not be (correctly) recognized by Yahoo Finance."
)

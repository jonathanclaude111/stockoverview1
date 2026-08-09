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
import anthropic
import json
import os

st.set_page_config(page_title="AI-Aandelen Dashboard", layout="wide")

CLAUDE_MODEL = "claude-sonnet-5"
TICKERS_FILE = os.path.join(os.path.dirname(__file__), "tickers.json")


def get_claude_client():
    """Geeft een Anthropic-client terug, of None als er geen API-key is ingesteld."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def extract_text(response) -> str:
    """Haalt de tekst uit een Claude-antwoord, ongeacht of er ook 'thinking'-blokken voor staan."""
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(parts) if parts else "⚠️ Geen tekstantwoord ontvangen van Claude."

DEFAULT_TICKERS = [
    "NVDA", "MU", "GOOGL", "TSLA", "MSFT", "AVGO", "META", "ARM", "QCOM", "AMD",
    "WDC", "AXTI", "APH", "ATEN", "ALAB", "APLD", "CIEN", "COHR", "CRDO", "DELL",
    "VRT", "NBIS", "PDD", "AAPL", "SOF.BR", "BABA", "LLY", "TSM", "PLTR", "CBRS",
    "CRWV", "QBTS", "MVIS", "ADUR", "POET", "IBM", "SPCX", "AMZN", "ORCL", "MRVL",
    "SNDK", "ANET", "SMCI", "AMAT", "LRCX", "KLAC", "INTC",
]


def load_tickers() -> list[str]:
    """Leest de tickerlijst van schijf, of geeft de standaardlijst terug als er nog geen bestand is."""
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
    """Schrijft de tickerlijst weg naar schijf zodat ze bewaard blijft na herladen."""
    try:
        with open(TICKERS_FILE, "w") as f:
            json.dump(tickers, f)
    except Exception as e:
        st.warning(f"Kon tickerlijst niet opslaan: {e}")


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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_claude_analysis(ticker: str, snapshot: dict) -> str:
    """Vraagt Claude om de cijfers te interpreteren en een koopwaardigheids-inschatting te geven.
    snapshot bevat enkel de kale cijfers (geen DataFrames) zodat het cachebaar/serialiseerbaar is."""
    client = get_claude_client()
    if client is None:
        return "⚠️ Geen ANTHROPIC_API_KEY ingesteld — zie instructies in de README."

    prompt = f"""Je bent een nuchtere, kritische aandelenanalist. Hieronder staan de actuele
fundamentele cijfers van {snapshot.get('name')} ({ticker}). Geef een korte, concrete analyse in het Nederlands:

1. Interpretatie van de cijfers (waardering, groei, gezondheid) — 3-4 zinnen
2. Argumenten die pleiten voor ondergewaardeerd zijn
3. Argumenten die pleiten voor overgewaardeerd zijn
4. Een voorzichtige inschatting van een aantrekkelijk instapniveau (prijs of multiple), met duidelijke
   vermelding dat dit geen financieel advies is, enkel een educatieve inschatting op basis van de cijfers

Cijfers:
- Koers: {snapshot.get('price')} {snapshot.get('currency')}
- Marktkap: {snapshot.get('market_cap')}
- P/E trailing: {snapshot.get('pe_trailing')} — P/E forward: {snapshot.get('pe_forward')}
- PEG-ratio: {snapshot.get('peg_ratio')}
- FCF (TTM): {snapshot.get('fcf_ttm')}
- 52-weekbereik: {snapshot.get('52w_low')} - {snapshot.get('52w_high')}
- YTD-rendement: {snapshot.get('ytd_return')}%
- Analisten koersdoel gemiddeld/laag/hoog: {snapshot.get('target_mean')} / {snapshot.get('target_low')} / {snapshot.get('target_high')}
- Analisten rating: {snapshot.get('recommendation')} (obv {snapshot.get('num_analysts')} analisten)
- Sector: {snapshot.get('sector')} — Industrie: {snapshot.get('industry')}

Houd het compact (max. ~250 woorden), gebruik korte alinea's met vetgedrukte tussenkopjes."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text(response)
    except Exception as e:
        return f"⚠️ Fout bij ophalen AI-analyse: {e}"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_top_picks(strong_buy_rows: list[dict]) -> str:
    """Vraagt Claude om, gegeven enkel de Strong Buy-aandelen, de 3 interessantste te kiezen."""
    client = get_claude_client()
    if client is None:
        return "⚠️ Geen ANTHROPIC_API_KEY ingesteld — zie instructies in de README."

    lines = []
    for r in strong_buy_rows:
        upside = None
        if r.get("price") and r.get("target_mean"):
            try:
                upside = (r["target_mean"] / r["price"] - 1) * 100
            except Exception:
                upside = None
        lines.append(
            f"- {r['ticker']} ({r['name']}): koers {r.get('price')}, "
            f"P/E trailing {r.get('pe_trailing')}, P/E forward {r.get('pe_forward')}, "
            f"PEG {r.get('peg_ratio')}, FCF TTM {r.get('fcf_ttm')}, YTD {r.get('ytd_return')}%, "
            f"koersdoel gem. {r.get('target_mean')} (upside ~{upside:.0f}% indien gekend)"
        )
    stock_list_text = "\n".join(lines)

    prompt = f"""Je bent een kritische, nuchtere aandelenanalist. Hieronder staan ALLE aandelen uit
mijn watchlist die momenteel een 'Strong Buy'-rating van analisten hebben. Kies hieruit de 3
interessantste om NU in te stappen, en leg uit waarom.

Belangrijk:
- Wees kritisch: negeer of markeer expliciet aandelen met negatieve winst/FCF, extreme P/E's,
  of onrealistische koersdoelen als (te) speculatief
- Verkies een goede balans tussen waardering (lage PEG/forward P/E), sterke FCF, en upside-potentieel
- Vermeld expliciet dat dit een educatieve, cijfermatige interpretatie is, geen financieel advies
- Antwoord in het Nederlands, compact, met voor elke top-3 keuze een korte onderbouwing (2-3 zinnen)

Aandelen:
{stock_list_text}"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text(response)
    except Exception as e:
        return f"⚠️ Fout bij ophalen AI-analyse: {e}"


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


# ----------------------- Ticker-beheer (sidebar) -----------------------

if "tickers" not in st.session_state:
    st.session_state.tickers = load_tickers()

with st.sidebar:
    st.header("⚙️ Aandelenlijst beheren")

    new_ticker = st.text_input("Ticker toevoegen (bv. NFLX)", key="new_ticker_input").strip().upper()
    if st.button("➕ Toevoegen"):
        if not new_ticker:
            st.warning("Vul eerst een ticker in.")
        elif new_ticker in st.session_state.tickers:
            st.warning(f"{new_ticker} staat al in de lijst.")
        else:
            st.session_state.tickers.append(new_ticker)
            save_tickers(st.session_state.tickers)
            st.success(f"{new_ticker} toegevoegd en opgeslagen.")
            st.rerun()

    st.markdown("---")
    to_remove = st.multiselect("Tickers verwijderen", options=st.session_state.tickers)
    if st.button("🗑️ Verwijder geselecteerde") and to_remove:
        st.session_state.tickers = [t for t in st.session_state.tickers if t not in to_remove]
        save_tickers(st.session_state.tickers)
        st.success(f"Verwijderd en opgeslagen: {', '.join(to_remove)}")
        st.rerun()

    st.markdown("---")
    if st.button("↩️ Reset naar standaardlijst"):
        st.session_state.tickers = DEFAULT_TICKERS.copy()
        save_tickers(st.session_state.tickers)
        st.rerun()

    st.caption(
        f"Huidige lijst: {len(st.session_state.tickers)} aandelen. Wijzigingen blijven bewaard "
        "na herladen, maar kunnen resetten na een herstart van de app op Streamlit Cloud. "
        "⚠️ Nieuwe tickers worden niet gevalideerd vooraf — als de ticker niet bestaat op "
        "Yahoo Finance, zie je 'n.b.' in de tabel in plaats van een foutmelding."
    )

TICKERS = st.session_state.tickers

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

    st.markdown("---")
    st.markdown("### 🏆 Top 3 Strong Buy-kandidaten")
    st.caption(
        "Filtert alle aandelen met 'Strong Buy'-rating en vraagt Claude om de 3 interessantste "
        "te selecteren op basis van waardering, FCF en upside. Geen financieel advies."
    )
    if st.button("Genereer top 3"):
        strong_buy_df = df[df["recommendation"].str.lower() == "strong_buy"]
        if strong_buy_df.empty:
            st.write("Geen aandelen met 'Strong Buy'-rating gevonden in de huidige data.")
        else:
            strong_buy_rows = strong_buy_df.drop(columns=["insider_transactions"], errors="ignore").to_dict("records")
            with st.spinner(f"Claude analyseert {len(strong_buy_rows)} Strong Buy-aandelen..."):
                top_picks = get_top_picks(strong_buy_rows)
            st.markdown(top_picks)

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

    st.markdown("---")
    st.markdown("### 🤖 AI-interpretatie & koopwaardigheid")
    st.caption(
        "Gegenereerd door Claude op basis van bovenstaande cijfers. "
        "Dit is geen financieel advies, enkel een educatieve inschatting."
    )
    if st.button("Genereer AI-analyse", key=f"analyse_{ticker_choice}"):
        # Enkel serialiseerbare kale waarden meegeven aan de cache-functie
        snapshot = {k: v for k, v in d.items() if k != "insider_transactions"}
        with st.spinner("Claude analyseert de cijfers..."):
            analysis = get_claude_analysis(ticker_choice, snapshot)
        st.markdown(analysis)

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

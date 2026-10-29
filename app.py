"""
AI Stock Dashboard
------------------
Streamlit-app die beursgegevens ophaalt via yfinance (Yahoo Finance)
en optioneel een AI-analyse maakt via de Claude API.

Lokaal draaien:
    pip install -r requirements.txt
    streamlit run app.py

Nodige "Secrets" in Streamlit Cloud (Manage app -> Settings -> Secrets):
    APP_PASSWORD = "een-lang-wachtwoord"
    ANTHROPIC_API_KEY = "sk-ant-..."   (optioneel, enkel voor de AI-knoppen)
    FMP_API_KEY = "..."                (optioneel, reservebron koersdoelen en forward P/E)
    FINNHUB_API_KEY = "..."            (optioneel, reservebron rating)
"""

import hmac
import json
import logging
import math
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

# Fouten worden in het logboek geschreven (zichtbaar in Streamlit Cloud via "Manage app").
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock_dashboard")

CLAUDE_MODEL = "claude-sonnet-5-5"
TICKERS_FILE = os.path.join(os.path.dirname(__file__), "tickers.json")
CACHE_TTL = 15 * 60  # 15 minuten
ANALYST_CACHE_TTL = 24 * 60 * 60  # analistendata verandert traag: 1 dag bewaren
BRUSSELS = ZoneInfo("Europe/Brussels")
MAX_TICKERS = 100
# Een ticker mag enkel letters, cijfers, punt, streepje, ^ of = bevatten (max 15 tekens).
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")

DEFAULT_TICKERS = [
    "NVDA", "MU", "GOOGL", "TSLA", "MSFT", "AVGO", "META", "ARM", "QCOM", "AMD",
    "WDC", "AXTI", "APH", "ATEN", "ALAB", "APLD", "CIEN", "COHR", "CRDO", "DELL",
    "VRT", "NBIS", "PDD", "AAPL", "SOF.BR", "BABA", "LLY", "TSM", "PLTR", "CBRS",
    "CRWV", "QBTS", "MVIS", "ADUR", "POET", "IBM", "SPCX", "AMZN", "ORCL", "MRVL",
    "SNDK", "ANET", "SMCI", "AMAT", "LRCX", "KLAC", "INTC",
]


# ----------------------- Hulpfuncties -----------------------

def md_escape(text) -> str:
    """Maakt tekst van buitenaf (Yahoo, gebruiker) veilig voor weergave in markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", str(text))


def get_secret(name: str):
    """Leest een geheim uit Streamlit Secrets. Geeft None als het niet bestaat."""
    try:
        return st.secrets.get(name)
    except Exception as e:  # bv. geen secrets-bestand aanwezig bij lokaal draaien
        logger.warning("Secret %s kon niet gelezen worden: %s", name, type(e).__name__)
        return None


def to_number(x):
    """Zet een waarde om naar een getal. Geeft None bij lege, foute of oneindige waarden."""
    if isinstance(x, bool) or not isinstance(x, (int, float, np.number)):
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def fmt_num(x, prefix="", suffix="", decimals=2):
    """Toont een getal netjes (bv. 1.23B), of 'n/a' als het ontbreekt."""
    x = to_number(x)
    if x is None:
        return "n/a"
    if abs(x) >= 1e12:
        return f"{prefix}{x/1e12:.{decimals}f}T{suffix}"
    if abs(x) >= 1e9:
        return f"{prefix}{x/1e9:.{decimals}f}B{suffix}"
    if abs(x) >= 1e6:
        return f"{prefix}{x/1e6:.{decimals}f}M{suffix}"
    return f"{prefix}{x:,.{decimals}f}{suffix}"


def validate_ticker(raw: str):
    """Controleert een ingetypte ticker. Geeft (ticker, None) of (None, foutmelding)."""
    ticker = (raw or "").strip().upper()
    if not ticker:
        return None, "Typ eerst een ticker, bv. NFLX."
    if not TICKER_PATTERN.match(ticker):
        return None, ("Ongeldige ticker. Gebruik enkel letters, cijfers, een punt of "
                      "een streepje (max 15 tekens), bv. NFLX of SOF.BR.")
    return ticker, None


# ----------------------- Wachtwoordbeveiliging -----------------------

def check_password() -> bool:
    """Toont een loginscherm. Geeft True als de gebruiker het juiste wachtwoord gaf."""
    if st.session_state.get("authenticated"):
        return True

    expected = get_secret("APP_PASSWORD")
    if not expected:
        # Veilig falen: zonder ingesteld wachtwoord blijft de app dicht.
        st.error("Geen wachtwoord ingesteld. Voeg APP_PASSWORD toe bij Secrets in "
                 "Streamlit Cloud (Manage app -> Settings -> Secrets).")
        return False

    st.title("🔒 AI Stock Dashboard")
    entered = st.text_input("Wachtwoord", type="password")
    if st.button("Inloggen"):
        # compare_digest vergelijkt veilig (geen hint via de rekentijd).
        if hmac.compare_digest(entered.encode("utf-8"), str(expected).encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Verkeerd wachtwoord.")
    return False


if not check_password():
    st.stop()


# ----------------------- Tickerlijst opslaan/laden -----------------------

def load_tickers() -> list[str]:
    """Leest de tickerlijst van schijf, of geeft de standaardlijst terug."""
    if not os.path.exists(TICKERS_FILE):
        return DEFAULT_TICKERS.copy()
    try:
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("tickers.json kon niet gelezen worden: %s", e)
        return DEFAULT_TICKERS.copy()
    if isinstance(data, list):
        valid = [t for t in data if isinstance(t, str) and TICKER_PATTERN.match(t)]
        if valid:
            return valid[:MAX_TICKERS]
    return DEFAULT_TICKERS.copy()


def save_tickers(tickers: list[str]) -> None:
    """Bewaart de tickerlijst op schijf."""
    try:
        with open(TICKERS_FILE, "w", encoding="utf-8") as f:
            json.dump(tickers, f)
    except OSError as e:
        logger.warning("tickers.json kon niet opgeslagen worden: %s", e)
        st.warning("De lijst kon niet bewaard worden. Ze blijft wel actief tot je de pagina herlaadt.")


# ----------------------- Data ophalen -----------------------

def get_fcf(t):
    """Vrije kasstroom: som van de laatste 4 kwartalen (TTM).
    Lukt dat niet, dan het laatste jaarcijfer. Geeft (bedrag, bron)."""
    try:
        q = t.quarterly_cashflow
        if q is not None and not q.empty and "Free Cash Flow" in q.index:
            last4 = q.loc["Free Cash Flow"].iloc[:4]
            if len(last4) == 4 and last4.notna().all():
                return float(last4.sum()), "TTM"
    except Exception as e:
        logger.info("Kwartaal-kasstroom niet beschikbaar: %s", e)
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
            value = to_number(cf.loc["Free Cash Flow"].iloc[0])
            if value is not None:
                return value, "laatste boekjaar"
    except Exception as e:
        logger.info("Jaarlijkse kasstroom niet beschikbaar: %s", e)
    return None, "n/a"


def get_trailing_eps(t):
    """Winst per aandeel over de laatste 4 kwartalen (TTM). Geeft None als dat niet lukt."""
    try:
        q = t.quarterly_income_stmt
        if q is not None and not q.empty and "Diluted EPS" in q.index:
            last4 = q.loc["Diluted EPS"].iloc[:4]
            if len(last4) == 4 and last4.notna().all():
                return float(last4.sum())
    except Exception as e:
        logger.info("Kwartaalwinst niet beschikbaar: %s", e)
    return None


def get_info(t, ticker: str) -> dict:
    """Haalt de 'info' op bij Yahoo. Yahoo blokkeert dit soms voor Streamlit Cloud
    (te veel verzoeken); dan komt er {} terug."""
    try:
        return t.info or {}
    except Exception as e:
        logger.warning("Geen info voor %s: %s", ticker, e)
        return {}


def get_fast_info(t, ticker: str) -> dict:
    """Reserveroute: koers, munt, beurswaarde en 52-weken via een ander Yahoo-adres."""
    result = {}
    try:
        fi = t.fast_info
    except Exception as e:
        logger.info("Geen fast_info voor %s: %s", ticker, e)
        return result
    for key in ("last_price", "currency", "market_cap", "year_low", "year_high"):
        try:
            result[key] = fi[key]
        except Exception as e:
            logger.info("fast_info %s ontbreekt voor %s: %s", key, ticker, e)
    return result


# ----------------------- Reservebronnen voor analistendata -----------------------
# Enkel actief als de sleutel bij Secrets staat (FMP_API_KEY / FINNHUB_API_KEY).

class SourceError(Exception):
    """Een reservebron gaf een fout. Fouten worden niet in de cache bewaard."""


def fetch_json(url: str, params: dict, source: str):
    """Vraagt JSON op bij een externe bron. Geeft SourceError bij een fout.
    De sleutel zit in params en komt nooit in het logboek."""
    try:
        r = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        raise SourceError(f"{source} niet bereikbaar: {type(e).__name__}") from e
    if r.status_code != 200:
        raise SourceError(f"{source} gaf status {r.status_code} voor {params.get('symbol')}")
    try:
        return r.json()
    except ValueError as e:
        raise SourceError(f"{source} gaf geen geldige JSON") from e


@st.cache_data(ttl=ANALYST_CACHE_TTL, show_spinner=False)
def get_fmp_targets(ticker: str) -> dict:
    """Koersdoelen (gemiddeld/laag/hoog) via Financial Modeling Prep."""
    key = get_secret("FMP_API_KEY")
    if not key:
        return {}
    data = fetch_json("https://financialmodelingprep.com/stable/price-target-consensus",
                      {"symbol": ticker, "apikey": key}, "FMP")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        row = data[0]
        return {"target_mean": to_number(row.get("targetConsensus")),
                "target_low": to_number(row.get("targetLow")),
                "target_high": to_number(row.get("targetHigh"))}
    return {}


@st.cache_data(ttl=ANALYST_CACHE_TTL, show_spinner=False)
def get_fmp_forward_eps(ticker: str):
    """Verwachte winst per aandeel voor het eerstvolgende boekjaar via FMP."""
    key = get_secret("FMP_API_KEY")
    if not key:
        return None
    data = fetch_json("https://financialmodelingprep.com/stable/analyst-estimates",
                      {"symbol": ticker, "period": "annual", "apikey": key}, "FMP")
    if not isinstance(data, list):
        return None
    today = datetime.now(BRUSSELS).strftime("%Y-%m-%d")
    future = [row for row in data if isinstance(row, dict)
              and str(row.get("date", "")) >= today and to_number(row.get("epsAvg")) is not None]
    if not future:
        return None
    nearest = min(future, key=lambda row: str(row["date"]))
    return to_number(nearest["epsAvg"])


@st.cache_data(ttl=ANALYST_CACHE_TTL, show_spinner=False)
def get_finnhub_rating(ticker: str) -> dict:
    """Analistenrating via Finnhub (aantal Strong Buy / Buy / Hold / Sell / Strong Sell)."""
    key = get_secret("FINNHUB_API_KEY")
    if not key:
        return {}
    data = fetch_json("https://finnhub.io/api/v1/stock/recommendation",
                      {"symbol": ticker, "token": key}, "Finnhub")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return {}
    latest = data[0]  # Finnhub zet de meest recente maand eerst
    counts = [to_number(latest.get(k)) or 0 for k in
              ("strongBuy", "buy", "hold", "sell", "strongSell")]
    total = sum(counts)
    if total == 0:
        return {}
    # Gemiddelde score zoals Yahoo: 1 = Strong Buy ... 5 = Strong Sell.
    score = sum((i + 1) * c for i, c in enumerate(counts)) / total
    if score <= 1.5:
        rec = "strong_buy"
    elif score <= 2.5:
        rec = "buy"
    elif score <= 3.5:
        rec = "hold"
    elif score <= 4.5:
        rec = "underperform"
    else:
        rec = "sell"
    return {"recommendation": rec, "num_analysts": total}


def fill_analyst_gaps(data: dict) -> None:
    """Vult ontbrekende koersdoelen, forward P/E en rating aan uit de reservebronnen."""
    sources = []
    try:
        if data["target_mean"] is None:
            targets = get_fmp_targets(data["ticker"])
            if targets.get("target_mean") is not None:
                data.update(targets)
                sources.append("koersdoelen: FMP")
        if data["pe_forward"] is None and data["price"]:
            eps = get_fmp_forward_eps(data["ticker"])
            if eps and eps > 0:
                data["pe_forward"] = data["price"] / eps
                sources.append("forward P/E: FMP")
    except SourceError as e:
        logger.warning("%s", e)
    try:
        if data["recommendation"] in (None, "n/a", "none"):
            rating = get_finnhub_rating(data["ticker"])
            if rating:
                data.update(rating)
                sources.append("rating: Finnhub")
    except SourceError as e:
        logger.warning("%s", e)
    data["analyst_source"] = ", ".join(sources) if sources else "Yahoo"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_stock_info(ticker: str) -> dict:
    """Haalt alle gegevens van één aandeel op. Ontbrekende velden worden 'n/a'."""
    data = {"ticker": ticker}
    t = yf.Ticker(ticker)
    info = get_info(t, ticker)
    # Een echte info bevat een naam of koers; enkel 'trailingPegRatio' betekent: geblokkeerd.
    data["info_ok"] = bool(info.get("longName") or info.get("shortName")
                           or info.get("regularMarketPrice"))
    fast = {} if data["info_ok"] else get_fast_info(t, ticker)

    data["name"] = info.get("longName") or info.get("shortName") or ticker
    data["sector"] = info.get("sector", "n/a")
    data["industry"] = info.get("industry", "n/a")
    data["summary"] = info.get("longBusinessSummary", "Geen beschrijving beschikbaar.")
    data["price"] = (to_number(info.get("currentPrice")) or to_number(info.get("regularMarketPrice"))
                     or to_number(fast.get("last_price")))
    data["currency"] = info.get("currency") or fast.get("currency") or ""
    data["market_cap"] = to_number(info.get("marketCap")) or to_number(fast.get("market_cap"))
    data["pe_trailing"] = to_number(info.get("trailingPE"))
    if data["pe_trailing"] is None and data["price"]:
        # Zelf berekenen: koers gedeeld door de winst per aandeel van de laatste 12 maanden.
        eps = get_trailing_eps(t)
        if eps and eps > 0:
            data["pe_trailing"] = data["price"] / eps
    data["pe_forward"] = to_number(info.get("forwardPE"))
    data["peg_ratio"] = to_number(info.get("pegRatio")) or to_number(info.get("trailingPegRatio"))
    data["52w_low"] = to_number(info.get("fiftyTwoWeekLow")) or to_number(fast.get("year_low"))
    data["52w_high"] = to_number(info.get("fiftyTwoWeekHigh")) or to_number(fast.get("year_high"))
    data["target_mean"] = to_number(info.get("targetMeanPrice"))
    data["target_low"] = to_number(info.get("targetLowPrice"))
    data["target_high"] = to_number(info.get("targetHighPrice"))
    data["recommendation"] = info.get("recommendationKey", "n/a")
    data["num_analysts"] = to_number(info.get("numberOfAnalystOpinions"))
    fill_analyst_gaps(data)

    data["fcf"], data["fcf_source"] = get_fcf(t)

    insiders = None
    try:
        ins = t.insider_transactions
        if ins is not None and not ins.empty:
            insiders = ins.head(5)
    except Exception as e:
        logger.info("Geen insider-data voor %s: %s", ticker, e)
    data["insider_transactions"] = insiders

    ytd_return = None
    try:
        hist = t.history(period="ytd")
        if not hist.empty:
            ytd_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
    except Exception as e:
        logger.info("Geen koershistoriek voor %s: %s", ticker, e)
    data["ytd_return"] = to_number(ytd_return)

    return data


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_all_stocks(tickers: list[str]):
    """Haalt alle aandelen op. Geeft de tabel én het tijdstip van ophalen (Belgische tijd)."""
    rows = [get_stock_info(tk) for tk in tickers]
    fetched_at = datetime.now(BRUSSELS).strftime("%d/%m/%Y %H:%M")
    return pd.DataFrame(rows), fetched_at


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_correlation_matrix(tickers: list[str]) -> pd.DataFrame:
    if len(tickers) < 2:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, period="6mo", interval="1d", progress=False)["Close"]
        returns = raw.pct_change().dropna(how="all")
        return returns.corr()
    except Exception as e:
        logger.warning("Correlatie mislukt: %s", e)
        return pd.DataFrame()


# ----------------------- Claude (AI-analyse) -----------------------

def get_claude_client():
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def extract_text(response) -> str:
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if parts:
        return "\n".join(parts)
    return "⚠️ Claude gaf geen tekst terug. Probeer opnieuw."


def ask_claude(prompt: str) -> str:
    client = get_claude_client()
    if client is None:
        return "⚠️ Geen ANTHROPIC_API_KEY ingesteld bij Secrets. Zie de README."
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text(response)
    except Exception as e:
        # Details enkel in het logboek, niet op het scherm.
        logger.error("Claude-oproep mislukt: %s", e)
        return ("⚠️ De AI-analyse lukte niet. Kijk in Streamlit Cloud bij "
                "Manage app -> logs voor de details.")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_claude_analysis(ticker: str, snapshot: dict) -> str:
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
- Free cash flow ({snapshot.get('fcf_source')}): {snapshot.get('fcf')}
- 52-week range: {snapshot.get('52w_low')} - {snapshot.get('52w_high')}
- YTD return: {snapshot.get('ytd_return')}%
- Analyst target price avg/low/high: {snapshot.get('target_mean')} / {snapshot.get('target_low')} / {snapshot.get('target_high')}
- Analyst rating: {snapshot.get('recommendation')} (based on {snapshot.get('num_analysts')} analysts)
- Sector: {snapshot.get('sector')} — Industry: {snapshot.get('industry')}

Keep it compact (max ~250 words), use short paragraphs with bold subheadings."""
    return ask_claude(prompt)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_top_picks(strong_buy_rows: list[dict]) -> str:
    lines = []
    for r in strong_buy_rows:
        price = to_number(r.get("price"))
        target = to_number(r.get("target_mean"))
        # Opgelost: vroeger crashte dit als het koersdoel ontbrak.
        if price and target:
            upside_text = f"upside ~{(target / price - 1) * 100:.0f}%"
        else:
            upside_text = "upside unknown"
        lines.append(
            f"- {r['ticker']} ({r['name']}): price {price}, "
            f"P/E trailing {r.get('pe_trailing')}, P/E forward {r.get('pe_forward')}, "
            f"PEG {r.get('peg_ratio')}, FCF ({r.get('fcf_source')}) {r.get('fcf')}, "
            f"YTD {r.get('ytd_return')}%, avg target price {target} ({upside_text})"
        )

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
{chr(10).join(lines)}"""
    return ask_claude(prompt)


def rating_badge(rec) -> str:
    mapping = {
        "strong_buy": "🟢 Strong Buy", "buy": "🟢 Buy", "hold": "🟡 Hold",
        "underperform": "🔴 Underperform", "sell": "🔴 Sell", "none": "n/a",
    }
    rec = rec if isinstance(rec, str) else "n/a"
    return mapping.get(rec.lower(), rec)


# ----------------------- Zijbalk: tickerlijst beheren -----------------------

if "tickers" not in st.session_state:
    st.session_state.tickers = load_tickers()

with st.sidebar:
    st.header("⚙️ Manage stock list")

    new_ticker_raw = st.text_input("Add ticker (e.g. NFLX)", key="new_ticker_input")
    if st.button("➕ Add"):
        ticker, error = validate_ticker(new_ticker_raw)
        if error:
            st.warning(error)
        elif ticker in st.session_state.tickers:
            st.warning(f"{md_escape(ticker)} staat al in de lijst.")
        elif len(st.session_state.tickers) >= MAX_TICKERS:
            st.warning(f"Maximum {MAX_TICKERS} aandelen. Verwijder er eerst een.")
        else:
            st.session_state.tickers.append(ticker)
            save_tickers(st.session_state.tickers)
            st.rerun()

    st.markdown("---")
    to_remove = st.multiselect("Remove tickers", options=st.session_state.tickers)
    if st.button("🗑️ Remove selected") and to_remove:
        st.session_state.tickers = [t for t in st.session_state.tickers if t not in to_remove]
        save_tickers(st.session_state.tickers)
        st.rerun()

    st.markdown("---")
    if st.button("↩️ Reset to default list"):
        st.session_state.tickers = DEFAULT_TICKERS.copy()
        save_tickers(st.session_state.tickers)
        st.rerun()

    st.caption(
        f"{len(st.session_state.tickers)} stocks. Changes made here are lost when "
        "Streamlit Cloud restarts. For permanent changes, edit DEFAULT_TICKERS in app.py on GitHub."
    )

    st.markdown("---")
    if st.button("🚪 Uitloggen"):
        st.session_state["authenticated"] = False
        st.rerun()

TICKERS = st.session_state.tickers

# ----------------------- Hoofdscherm -----------------------

st.title("📊 AI Stock Dashboard")

with st.spinner("Fetching data for all stocks... (may take a moment on first load)"):
    df, fetched_at = get_all_stocks(TICKERS)

st.caption(f"Data via Yahoo Finance, opgehaald op {fetched_at} (Belgische tijd). "
           "Wordt elke 15 minuten vernieuwd. Koersen kunnen ~15 min vertraagd zijn.")

blocked = int((~df["info_ok"]).sum()) if "info_ok" in df else 0
if blocked:
    st.warning(
        f"Yahoo Finance gaf voor {blocked} van de {len(df)} aandelen niet alle gegevens "
        "(Yahoo beperkt soms het aantal verzoeken). Koers, beurswaarde en P/E (trailing) "
        "worden dan via een reserveroute opgehaald. Forward P/E, koersdoelen en rating "
        "komen uit FMP en Finnhub als je die sleutels bij Secrets hebt ingesteld, anders "
        "blijven ze n/a. Probeer het over enkele minuten opnieuw."
    )
    if st.button("🔄 Opnieuw ophalen"):
        # Enkel Yahoo opnieuw; de reservebronnen hebben een daglimiet.
        get_stock_info.clear()
        get_all_stocks.clear()
        st.rerun()

tab_overview, tab_detail, tab_corr = st.tabs(["📋 Overview", "🔍 Stock detail", "🔗 Correlation"])

with tab_overview:
    st.subheader("Overview table")
    display_df = df.copy()
    display_df["Price"] = display_df.apply(lambda r: fmt_num(r["price"], suffix=f" {r['currency']}"), axis=1)
    display_df["Market Cap"] = display_df["market_cap"].apply(fmt_num)
    display_df["P/E (trailing)"] = display_df["pe_trailing"].apply(fmt_num)
    display_df["P/E (forward)"] = display_df["pe_forward"].apply(fmt_num)
    display_df["PEG"] = display_df["peg_ratio"].apply(fmt_num)
    display_df["FCF"] = display_df["fcf"].apply(fmt_num)
    display_df["FCF basis"] = display_df["fcf_source"]
    display_df["YTD %"] = display_df["ytd_return"].apply(lambda x: fmt_num(x, suffix="%"))
    display_df["Avg. Target"] = display_df["target_mean"].apply(fmt_num)
    display_df["Rating"] = display_df["recommendation"].apply(rating_badge)

    cols_to_show = [
        "ticker", "name", "Price", "Market Cap", "P/E (trailing)", "P/E (forward)",
        "PEG", "FCF", "FCF basis", "YTD %", "Avg. Target", "Rating",
    ]
    st.dataframe(
        display_df[cols_to_show].rename(columns={"ticker": "Ticker", "name": "Name"}),
        width="stretch", hide_index=True, height=600,
    )

    st.markdown("---")
    st.markdown("### 🏆 Top 3 Strong Buy candidates")
    st.caption("Claude picks the 3 most interesting 'Strong Buy' stocks. Not financial advice.")
    if st.button("Generate top 3"):
        strong_buy_df = df[df["recommendation"].astype(str).str.lower() == "strong_buy"]
        if strong_buy_df.empty:
            st.write("No stocks with a 'Strong Buy' rating found in the current data.")
        else:
            rows = strong_buy_df.drop(columns=["insider_transactions", "info_ok", "analyst_source"], errors="ignore").to_dict("records")
            with st.spinner(f"Claude is analyzing {len(rows)} Strong Buy stocks..."):
                st.session_state["top_picks_text"] = get_top_picks(rows)

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

    st.markdown(f"### {md_escape(d['name'])} ({md_escape(d['ticker'])})")
    st.markdown(f"**Sector:** {md_escape(d['sector'])} — **Industry:** {md_escape(d['industry'])}")
    st.markdown("**Business description:**")
    st.markdown(md_escape(d["summary"]))

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P/E trailing", fmt_num(d["pe_trailing"]))
    c2.metric("P/E forward", fmt_num(d["pe_forward"]))
    c3.metric("PEG ratio", fmt_num(d["peg_ratio"]))
    c4.metric(f"FCF ({d['fcf_source']})", fmt_num(d["fcf"]))

    c5, c6 = st.columns(2)
    c5.metric("52-week range", f"{fmt_num(d['52w_low'])} — {fmt_num(d['52w_high'])}")
    c6.metric("Analyst rating", rating_badge(d["recommendation"]))

    analysts = fmt_num(d["num_analysts"], decimals=0)
    st.markdown("**Analyst price targets:**")
    st.write(
        f"Average: {fmt_num(d['target_mean'])} — Low: {fmt_num(d['target_low'])} — "
        f"High: {fmt_num(d['target_high'])} (based on {analysts} analysts)"
    )
    st.caption(f"Bron analistendata: {d['analyst_source']}")

    st.markdown("**Recent insider transactions:**")
    if d["insider_transactions"] is not None and not d["insider_transactions"].empty:
        st.dataframe(d["insider_transactions"], width="stretch", hide_index=True)
    else:
        st.write("No data available.")

    st.markdown("---")
    st.markdown("### 🤖 AI interpretation & buy-worthiness")
    st.caption("Generated by Claude based on the figures above. Not financial advice.")
    analysis_key = f"analysis_text_{ticker_choice}"
    if st.button("Generate AI analysis", key=f"analyse_{ticker_choice}"):
        snapshot = {k: v for k, v in d.items() if k not in ("insider_transactions", "info_ok", "analyst_source")}
        with st.spinner("Claude is analyzing the figures..."):
            st.session_state[analysis_key] = get_claude_analysis(ticker_choice, snapshot)

    if analysis_key in st.session_state:
        st.markdown(st.session_state[analysis_key])

with tab_corr:
    st.subheader("Correlation (based on 6 months of daily returns)")
    with st.spinner("Calculating correlation matrix..."):
        corr = get_correlation_matrix(TICKERS)
    if not corr.empty:
        st.dataframe(
            corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
            width="stretch", height=700,
        )
    else:
        st.write("Could not calculate the correlation matrix.")

st.markdown("---")
st.caption(
    "⚠️ This data is for informational purposes only and does not constitute investment advice. "
    "Some tickers may not be (correctly) recognized by Yahoo Finance."
)

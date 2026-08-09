# AI Stock Dashboard

A Streamlit web app showing live fundamental data for a list of AI-related stocks,
powered by Yahoo Finance (`yfinance`) — no API key needed for the market data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the link shown in your terminal (usually `http://localhost:8501`).

## Hosting via Streamlit Community Cloud (free)

1. Create a free account at https://share.streamlit.io (sign in with GitHub)
2. Put this folder (`app.py` + `requirements.txt`) in a new GitHub repository
3. Click "New app" in Streamlit Cloud, pick your repository and `app.py` as the main file
4. Done — you'll get a public URL you can visit and share

## AI analysis (optional, via Claude API)

The app can generate an AI-written interpretation and buy-worthiness assessment per stock,
plus a "Top 3 Strong Buy" comparison, using the Anthropic API. To enable this:

1. Create an account at https://console.anthropic.com and generate an API key
2. In Streamlit Cloud: "Manage app" -> Settings -> Secrets, add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
3. Never commit your API key to GitHub or put it directly in app.py

Note: this uses paid API credits (typically a few cents per generated analysis) - separate
from your regular claude.ai subscription.

## Known caveat

Some tickers in the list (e.g. SOF.BR, CBRS, ADUR, SPCX) may not be fully recognized
by Yahoo Finance -- the app shows "n/a" for missing fields instead of crashing.

## Next steps (optional)

- Reddit sentiment (would require a Reddit API + sentiment analysis -- a separate project)
- Your own custom "buy-worthy level" scoring logic

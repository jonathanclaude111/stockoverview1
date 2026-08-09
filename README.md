# AI-Aandelen Dashboard

Een Streamlit-webapp die live fundamentele data toont voor een lijst AI-gerelateerde aandelen,
via Yahoo Finance (`yfinance`) — geen API-key nodig.

## Lokaal testen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open daarna de link die in je terminal verschijnt (meestal `http://localhost:8501`).

## Hosten via Streamlit Community Cloud (gratis)

1. Maak een gratis account op https://share.streamlit.io (inloggen met GitHub)
2. Zet deze map (`app.py` + `requirements.txt`) in een nieuwe GitHub-repository
3. Klik in Streamlit Cloud op "New app", kies je repository en het bestand `app.py`
4. Klaar — je krijgt een publieke URL die je kan bezoeken en delen

## Bekend aandachtspunt

Sommige tickers uit de lijst (bv. `SOF.BR`, `CBRS`, `ADUR`, `SPCX`) worden mogelijk niet
of niet volledig herkend door Yahoo Finance — de app toont dan "n.b." voor de ontbrekende
velden in plaats van te crashen. Check dit na de eerste run en pas zo nodig de tickerlijst
bovenaan in `app.py` aan.

## Volgende stappen (optioneel, later toe te voegen)

- AI-duiding per aandeel via de Claude API (bv. korte samenvatting van sterktes/zwaktes)
- Reddit-sentiment (vereist Reddit API + sentiment-analyse — apart project)
- Eigen "koopwaardig niveau"-logica op basis van jouw eigen criteria

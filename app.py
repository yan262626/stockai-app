"""
StockAI Pro v5.0 — Version Goldman Sachs Gratuite
Nouvelles sources :
- Options Flow (Put/Call ratio, Open Interest)
- Insider Trading SEC EDGAR (Form 4)
- Short Interest FINRA
- Momentum relatif sectoriel
- Earnings surprises historiques
- Corrélations sectorielles
"""

import os, json, time, re, numpy as np, requests
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import pandas as pd
    import yfinance as yf
    import xgboost as xgb
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except Exception:
    TRENDS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
GROQ_KEY    = os.environ.get("GROQ_KEY", "")
MODEL_CACHE    = {}
CACHE_DURATION = 3600
MAX_CACHE_SIZE = 10


def cache_valid(ticker):
    if ticker not in MODEL_CACHE:
        return False
    return (time.time() - MODEL_CACHE[ticker].get("trained_at", 0)) < CACHE_DURATION


def auto_clean_cache():
    if len(MODEL_CACHE) >= MAX_CACHE_SIZE:
        oldest = sorted(MODEL_CACHE.items(), key=lambda x: x[1].get("trained_at", 0))
        for ticker, _ in oldest[:3]:
            del MODEL_CACHE[ticker]


# ══ RÉSOLUTION TICKERS ═══════════════════════════════════════════════════════

KNOWN_TICKERS = {
    "apple":"AAPL","tesla":"TSLA","microsoft":"MSFT","google":"GOOGL",
    "alphabet":"GOOGL","amazon":"AMZN","meta":"META","facebook":"META",
    "nvidia":"NVDA","netflix":"NFLX","intel":"INTC","amd":"AMD",
    "salesforce":"CRM","oracle":"ORCL","ibm":"IBM","qualcomm":"QCOM",
    "broadcom":"AVGO","paypal":"PYPL","uber":"UBER","airbnb":"ABNB",
    "spotify":"SPOT","jpmorgan":"JPM","jp morgan":"JPM","goldman":"GS",
    "goldman sachs":"GS","morgan stanley":"MS","bank of america":"BAC",
    "citigroup":"C","wells fargo":"WFC","blackrock":"BLK","visa":"V",
    "mastercard":"MA","boeing":"BA","johnson":"JNJ","pfizer":"PFE",
    "moderna":"MRNA","merck":"MRK","walmart":"WMT","nike":"NKE",
    "coca cola":"KO","pepsi":"PEP","mcdonalds":"MCD","starbucks":"SBUX",
    "disney":"DIS","exxon":"XOM","chevron":"CVX","berkshire":"BRK-B",
    "lvmh":"MC.PA","total":"TTE.PA","totalenergies":"TTE.PA",
    "loreal":"OR.PA","l'oreal":"OR.PA","air liquide":"AI.PA",
    "sanofi":"SAN.PA","bnp":"BNP.PA","bnp paribas":"BNP.PA",
    "societe generale":"GLE.PA","airbus":"AIR.PA","safran":"SAF.PA",
    "hermes":"RMS.PA","carrefour":"CA.PA","orange":"ORA.PA",
    "engie":"ENGI.PA","capgemini":"CAP.PA",
    "iren":"IREN","marathon":"MARA","riot":"RIOT",
    "coinbase":"COIN","microstrategy":"MSTR",
}


def resolve_ticker(query):
    q = query.strip()
    if len(q)==12 and q[:2].isalpha() and q[2:].isalnum():
        resolved = resolve_isin(q)
        if resolved != q: return resolved
    if yf_test(q.upper()): return q.upper()
    q_lower = q.lower().strip()
    if q_lower in KNOWN_TICKERS: return KNOWN_TICKERS[q_lower]
    for name, ticker in KNOWN_TICKERS.items():
        if q_lower in name or name in q_lower: return ticker
    base = q.upper().replace(" ","")
    for suffix in ["PA","DE","L","MI","AS","MC","TO"]:
        candidate = f"{base}.{suffix}"
        if yf_test(candidate): return candidate
    if FINNHUB_KEY:
        try:
            r = requests.get(
                f"https://finnhub.io/api/v1/search?q={q}&token={FINNHUB_KEY}",
                timeout=5)
            for res in r.json().get("result",[])[:5]:
                if res.get("type") in ["Common Stock","EQS","ADR"]:
                    return res["symbol"]
        except: pass
    return q.upper()


def yf_test(ticker):
    try:
        return not yf.Ticker(ticker).history(period="5d").empty
    except: return False


def resolve_isin(isin):
    try:
        r = requests.post(
            "https://api.openfigi.com/v3/mapping",
            headers={"Content-Type":"application/json"},
            json=[{"idType":"ID_ISIN","idValue":isin}], timeout=8)
        return r.json()[0]["data"][0]["ticker"]
    except: return isin


# ══ SCORES ════════════════════════════════════════════════════════════════════

def normalize_score(value, min_val, max_val, inverse=False):
    if value is None: return 50
    try:
        score = (float(value)-min_val)/(max_val-min_val)*100
        score = max(0, min(100, score))
        return round(100-score if inverse else score, 1)
    except: return 50


def compute_vix_score(vix):
    return normalize_score(vix, 10, 45, inverse=True)


def compute_macro_score(sp500_5d, nasdaq_5d, gold_5d, dollar_5d, rate_10y):
    scores = []
    if sp500_5d is not None: scores.append(normalize_score(sp500_5d,-5,5))
    if nasdaq_5d is not None: scores.append(normalize_score(nasdaq_5d,-7,7))
    if gold_5d is not None: scores.append(normalize_score(gold_5d,-3,3,inverse=True))
    if rate_10y is not None: scores.append(normalize_score(rate_10y,1,5,inverse=True))
    return round(sum(scores)/len(scores),1) if scores else 50


def compute_fundamental_score(info):
    scores = []
    pe = info.get("peRatio")
    if pe and pe>0: scores.append(normalize_score(pe,5,50,inverse=True))
    roe = info.get("roe")
    if roe is not None: scores.append(normalize_score(roe*100,-10,30))
    margin = info.get("netMargin")
    if margin is not None: scores.append(normalize_score(margin*100,-5,25))
    rev = info.get("revenueGrowth")
    if rev is not None: scores.append(normalize_score(rev*100,-10,30))
    debt = info.get("debtToEquity")
    if debt is not None: scores.append(normalize_score(debt,0,200,inverse=True))
    fcf = info.get("freeCashflow")
    if fcf and fcf > 0: scores.append(75)
    elif fcf and fcf < 0: scores.append(25)
    return round(sum(scores)/len(scores),1) if scores else 50


def compute_analyst_score(info, consensus):
    scores = []
    current = info.get("currentPrice",0)
    target  = info.get("targetMeanPrice")
    if current and target and current>0:
        scores.append(normalize_score((target-current)/current*100,-20,50))
    tot = sum(consensus.values())
    if tot>0:
        scores.append((consensus.get("strongBuy",0)*2+consensus.get("buy",0))/(tot*2)*100)
    rec_map = {"strongBuy":95,"buy":75,"hold":50,"sell":25,"strongSell":5}
    rec = info.get("recommendationKey","")
    if rec in rec_map: scores.append(rec_map[rec])
    return round(sum(scores)/len(scores),1) if scores else 50


# ══ OPTIONS FLOW (NOUVEAU) ════════════════════════════════════════════════════

def fetch_options_flow(symbol):
    """
    Analyse le marché des options :
    - Put/Call ratio (>1.2 = bears dominants, <0.7 = bulls dominants)
    - Open Interest inhabituellement élevé = signal fort
    - Implied Volatility = anticipation de mouvement
    Signal utilisé par tous les hedge funds
    """
    result = {
        "put_call_ratio":    None,
        "signal":            "neutre",
        "score":             50,
        "total_call_volume": None,
        "total_put_volume":  None,
        "avg_call_iv":       None,
        "avg_put_iv":        None,
        "max_pain":          None,
        "unusual_activity":  False,
        "label":             "Pas de données options",
    }
    try:
        tk = yf.Ticker(symbol)
        exp_dates = tk.options
        if not exp_dates:
            return result

        # Prendre les 2 prochaines expirations (plus liquides)
        total_calls = 0; total_puts = 0
        call_iv_list = []; put_iv_list = []
        call_oi = 0; put_oi = 0

        for exp in exp_dates[:3]:
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls
                puts  = chain.puts

                total_calls += int(calls["volume"].fillna(0).sum())
                total_puts  += int(puts["volume"].fillna(0).sum())
                call_oi     += int(calls["openInterest"].fillna(0).sum())
                put_oi      += int(puts["openInterest"].fillna(0).sum())

                if "impliedVolatility" in calls.columns:
                    iv_calls = calls["impliedVolatility"].dropna()
                    if len(iv_calls): call_iv_list.append(float(iv_calls.mean()))
                if "impliedVolatility" in puts.columns:
                    iv_puts = puts["impliedVolatility"].dropna()
                    if len(iv_puts): put_iv_list.append(float(iv_puts.mean()))
            except: continue

        if total_calls + total_puts == 0:
            return result

        pc_ratio = total_puts / max(total_calls, 1)
        result["put_call_ratio"]    = round(pc_ratio, 3)
        result["total_call_volume"] = total_calls
        result["total_put_volume"]  = total_puts
        result["avg_call_iv"]       = round(float(np.mean(call_iv_list))*100, 1) if call_iv_list else None
        result["avg_put_iv"]        = round(float(np.mean(put_iv_list))*100, 1)  if put_iv_list  else None

        # Volume inhabituellement élevé par rapport à l'OI
        if call_oi > 0:
            vol_oi_ratio = total_calls / call_oi
            result["unusual_activity"] = vol_oi_ratio > 2.0

        # Signal et score
        if pc_ratio < 0.5:
            result["signal"] = "fortement_haussier"
            result["score"]  = 85
            result["label"]  = "Forte pression acheteuse sur options"
        elif pc_ratio < 0.7:
            result["signal"] = "haussier"
            result["score"]  = 68
            result["label"]  = "Biais haussier sur options"
        elif pc_ratio < 0.9:
            result["signal"] = "legerement_haussier"
            result["score"]  = 57
            result["label"]  = "Legerement haussier"
        elif pc_ratio < 1.1:
            result["signal"] = "neutre"
            result["score"]  = 50
            result["label"]  = "Marche options neutre"
        elif pc_ratio < 1.3:
            result["signal"] = "legerement_baissier"
            result["score"]  = 43
            result["label"]  = "Legerement baissier"
        elif pc_ratio < 1.6:
            result["signal"] = "baissier"
            result["score"]  = 32
            result["label"]  = "Pression vendeuse sur options"
        else:
            result["signal"] = "fortement_baissier"
            result["score"]  = 18
            result["label"]  = "Forte protection baissiere"

        if result["unusual_activity"]:
            result["label"] += " + ACTIVITE INHABITUELLE"

    except Exception as e:
        print(f"Options flow erreur: {e}")
    return result


# ══ INSIDER TRADING SEC EDGAR (NOUVEAU) ═══════════════════════════════════════

def fetch_insider_trading(symbol):
    """
    Récupère les transactions des insiders depuis SEC EDGAR (Form 4).
    Les achats des dirigeants = signal haussier très fort.
    Les ventes massives = signal baissier.
    100% gratuit et public.
    """
    result = {
        "transactions":    [],
        "net_shares":      0,
        "buy_value":       0,
        "sell_value":      0,
        "score":           50,
        "signal":          "neutre",
        "label":           "Pas de transactions recentes",
        "buyers":          0,
        "sellers":         0,
    }
    try:
        # SEC EDGAR full-text search
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{symbol}"',
            "dateRange": "custom",
            "startdt": (date.today()-timedelta(days=90)).strftime("%Y-%m-%d"),
            "enddt": date.today().strftime("%Y-%m-%d"),
            "forms": "4",
        }
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": "StockAI research@stockai.com"})
        if r.status_code != 200:
            return result

        data = r.json()
        hits = data.get("hits", {}).get("hits", [])

        transactions = []
        for hit in hits[:20]:
            src = hit.get("_source", {})
            display_names = src.get("display_names", [])
            if not display_names: continue
            transactions.append({
                "insider":   display_names[0] if display_names else "Unknown",
                "date":      src.get("file_date",""),
                "form":      src.get("form_type","4"),
            })

        result["transactions"] = transactions[:8]

        # Finnhub insider transactions (plus fiable si clé disponible)
        if FINNHUB_KEY:
            try:
                from_date = (date.today()-timedelta(days=90)).strftime("%Y-%m-%d")
                to_date   = date.today().strftime("%Y-%m-%d")
                r2 = requests.get(
                    f"https://finnhub.io/api/v1/stock/insider-transactions"
                    f"?symbol={symbol}&from={from_date}&to={to_date}"
                    f"&token={FINNHUB_KEY}", timeout=8)
                insider_data = r2.json().get("data", [])

                buy_val=0; sell_val=0; net=0; buyers=set(); sellers=set()
                for tx in insider_data[:30]:
                    shares = tx.get("share", 0) or 0
                    price  = tx.get("transactionPrice", 0) or 0
                    ttype  = tx.get("transactionCode", "")
                    name   = tx.get("name","")
                    value  = abs(shares * price)

                    if ttype in ["P","A"]:  # Purchase, Award
                        buy_val += value; net += shares; buyers.add(name)
                    elif ttype in ["S","D"]:  # Sale, Disposition
                        sell_val += value; net -= shares; sellers.add(name)

                result["buy_value"]  = round(buy_val, 0)
                result["sell_value"] = round(sell_val, 0)
                result["net_shares"] = round(net, 0)
                result["buyers"]     = len(buyers)
                result["sellers"]    = len(sellers)

                # Score et signal
                if buy_val > 0 or sell_val > 0:
                    ratio = buy_val / max(buy_val + sell_val, 1)
                    if ratio > 0.8 and len(buyers) >= 2:
                        result["signal"] = "fortement_haussier"
                        result["score"]  = 88
                        result["label"]  = f"{len(buyers)} insiders acheteurs — signal tres fort"
                    elif ratio > 0.6:
                        result["signal"] = "haussier"
                        result["score"]  = 70
                        result["label"]  = f"Insiders acheteurs (ratio achat {round(ratio*100)}%)"
                    elif ratio < 0.2 and len(sellers) >= 2:
                        result["signal"] = "baissier"
                        result["score"]  = 25
                        result["label"]  = f"{len(sellers)} insiders vendeurs — attention"
                    elif ratio < 0.4:
                        result["signal"] = "legerement_baissier"
                        result["score"]  = 38
                        result["label"]  = "Plus de ventes que d'achats insiders"
                    else:
                        result["signal"] = "neutre"
                        result["score"]  = 50
                        result["label"]  = "Activite insiders equilibree"

            except Exception as e:
                print(f"Finnhub insider: {e}")

    except Exception as e:
        print(f"Insider trading erreur: {e}")
    return result


# ══ SHORT INTEREST (NOUVEAU) ═══════════════════════════════════════════════════

def fetch_short_interest(symbol, info):
    """
    Analyse le short interest :
    - Short % of float élevé + volume fort = risque de short squeeze
    - Court-circuitage massif = signal baissier des professionnels
    """
    result = {
        "short_pct_float": None,
        "short_ratio":     None,
        "days_to_cover":   None,
        "squeeze_risk":    "faible",
        "score":           50,
        "signal":          "neutre",
        "label":           "Donnees short indisponibles",
    }
    try:
        short_pct   = info.get("shortPct")
        short_ratio = info.get("shortRatio")

        if short_pct:
            result["short_pct_float"] = round(float(short_pct)*100, 2)
        if short_ratio:
            result["short_ratio"]     = round(float(short_ratio), 1)
            result["days_to_cover"]   = round(float(short_ratio), 1)

        spct = result["short_pct_float"]
        if spct is None:
            return result

        # Évaluation du risque de short squeeze
        if spct > 30:
            result["squeeze_risk"] = "extreme"
            result["score"]        = 72  # Potentiel de squeeze haussier
            result["signal"]       = "squeeze_risk_extreme"
            result["label"]        = f"Short squeeze possible — {spct}% vendu a decouvert"
        elif spct > 20:
            result["squeeze_risk"] = "eleve"
            result["score"]        = 63
            result["signal"]       = "squeeze_risk_eleve"
            result["label"]        = f"Risque squeeze eleve — {spct}% short"
        elif spct > 10:
            result["squeeze_risk"] = "modere"
            result["score"]        = 45
            result["signal"]       = "pression_baissiere_modere"
            result["label"]        = f"Pression baissiere moderee — {spct}% short"
        elif spct > 5:
            result["squeeze_risk"] = "normal"
            result["score"]        = 52
            result["signal"]       = "neutre"
            result["label"]        = f"Short interest normal — {spct}%"
        else:
            result["squeeze_risk"] = "faible"
            result["score"]        = 58
            result["signal"]       = "peu_shorte"
            result["label"]        = f"Peu vendu a decouvert — {spct}% (confiance)"

    except Exception as e:
        print(f"Short interest erreur: {e}")
    return result


# ══ MOMENTUM RELATIF SECTORIEL (NOUVEAU) ════════════════════════════════════

def fetch_sector_momentum(symbol, info):
    """
    Compare la performance de l'action vs son secteur et le SP500.
    Une action qui surperforme son secteur est en position de force.
    """
    result = {
        "vs_sp500_1m":    None,
        "vs_sp500_3m":    None,
        "vs_sector_1m":   None,
        "sector_etf":     None,
        "relative_strength": 50,
        "signal":         "neutre",
        "label":          "Calcul en cours",
    }

    # Mapping secteur → ETF sectoriel (gratuit yfinance)
    sector_etf_map = {
        "Technology":           "XLK",
        "Financial Services":   "XLF",
        "Healthcare":           "XLV",
        "Consumer Cyclical":    "XLY",
        "Consumer Defensive":   "XLP",
        "Energy":               "XLE",
        "Industrials":          "XLI",
        "Basic Materials":      "XLB",
        "Real Estate":          "XLRE",
        "Utilities":            "XLU",
        "Communication Services":"XLC",
    }

    try:
        sector = info.get("sector","")
        etf    = sector_etf_map.get(sector)
        result["sector_etf"] = etf

        tickers_to_fetch = [symbol, "^GSPC"]
        if etf: tickers_to_fetch.append(etf)

        perf = {}
        for t in tickers_to_fetch:
            try:
                hist = yf.Ticker(t).history(period="3mo")
                if hist.empty: continue
                prices = hist["Close"].tolist()
                if len(prices) >= 22:
                    perf[t] = {
                        "1m": (prices[-1]/prices[-22]-1)*100,
                        "3m": (prices[-1]/prices[0]-1)*100,
                    }
            except: continue

        sp = perf.get("^GSPC",{})
        tk = perf.get(symbol,{})

        if sp and tk:
            result["vs_sp500_1m"] = round(tk.get("1m",0) - sp.get("1m",0), 2)
            result["vs_sp500_3m"] = round(tk.get("3m",0) - sp.get("3m",0), 2)

        if etf and etf in perf and tk:
            result["vs_sector_1m"] = round(tk.get("1m",0) - perf[etf].get("1m",0), 2)

        # Score de force relative
        scores = []
        if result["vs_sp500_1m"] is not None:
            scores.append(normalize_score(result["vs_sp500_1m"], -10, 10))
        if result["vs_sp500_3m"] is not None:
            scores.append(normalize_score(result["vs_sp500_3m"], -15, 15))
        if result["vs_sector_1m"] is not None:
            scores.append(normalize_score(result["vs_sector_1m"], -8, 8))

        if scores:
            rs = round(sum(scores)/len(scores), 1)
            result["relative_strength"] = rs
            if rs >= 70:
                result["signal"] = "tres_fort"
                result["label"]  = f"Surperforme marche et secteur — force exceptionnelle"
            elif rs >= 60:
                result["signal"] = "fort"
                result["label"]  = f"Au-dessus du marche — momentum positif"
            elif rs >= 45:
                result["signal"] = "neutre"
                result["label"]  = "Performe comme le marche"
            elif rs >= 35:
                result["signal"] = "faible"
                result["label"]  = "Sous-performe le marche"
            else:
                result["signal"] = "tres_faible"
                result["label"]  = "Fortement sous le marche — momentum negatif"

    except Exception as e:
        print(f"Sector momentum erreur: {e}")
    return result


# ══ EARNINGS SURPRISES HISTORIQUES (NOUVEAU) ═════════════════════════════════

def fetch_earnings_quality(symbol):
    """
    Analyse la qualité des résultats sur les 8 derniers trimestres :
    - Combien de fois a battu les estimations ?
    - Amplitude des surprises
    - Tendance des révisions
    Signal fondamental très utilisé par les gérants
    """
    result = {
        "beat_rate":      None,
        "avg_surprise":   None,
        "last_4_beats":   None,
        "trend":          "neutre",
        "score":          50,
        "label":          "Historique earnings indisponible",
        "quarters":       [],
    }

    if not FINNHUB_KEY:
        return result

    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/stock/earnings"
            f"?symbol={symbol}&token={FINNHUB_KEY}", timeout=8)
        earnings = r.json()

        if not earnings:
            return result

        quarters = []
        for e in earnings[:8]:
            actual   = e.get("actual")
            estimate = e.get("estimate")
            if actual is None or estimate is None: continue
            surprise_pct = (actual-estimate)/abs(estimate+1e-9)*100
            beat = actual >= estimate
            quarters.append({
                "period":       e.get("period",""),
                "actual":       actual,
                "estimate":     estimate,
                "surprise_pct": round(surprise_pct, 1),
                "beat":         beat,
            })

        if not quarters:
            return result

        result["quarters"]   = quarters[:4]
        beat_count           = sum(1 for q in quarters if q["beat"])
        result["beat_rate"]  = round(beat_count/len(quarters)*100, 0)
        result["avg_surprise"]= round(float(np.mean([q["surprise_pct"] for q in quarters])), 1)
        result["last_4_beats"]= sum(1 for q in quarters[:4] if q["beat"])

        # Tendance : est-ce que les surprises s'améliorent ?
        if len(quarters) >= 4:
            recent_avg = np.mean([q["surprise_pct"] for q in quarters[:2]])
            older_avg  = np.mean([q["surprise_pct"] for q in quarters[2:4]])
            result["trend"] = "amelioration" if recent_avg > older_avg else "deterioration"

        # Score
        br = result["beat_rate"]
        avg_surp = result["avg_surprise"]
        score = normalize_score(br, 30, 90)
        if avg_surp > 10: score = min(100, score+10)
        elif avg_surp < 0: score = max(0, score-10)
        if result["trend"] == "amelioration": score = min(100, score+5)
        result["score"] = round(score, 1)

        if br >= 80 and avg_surp > 5:
            result["label"] = f"Beats estimations {int(br)}% du temps — qualite exceptionelle"
        elif br >= 70:
            result["label"] = f"Beats estimations regulierement ({int(br)}%)"
        elif br >= 50:
            result["label"] = f"Resultats mixtes — beats {int(br)}% du temps"
        else:
            result["label"] = f"Deceptions frequentes — beats seulement {int(br)}%"

    except Exception as e:
        print(f"Earnings quality erreur: {e}")
    return result


# ══ REDDIT SENTIMENT ════════════════════════════════════════════════════════

def fetch_reddit_sentiment(ticker, company_name=""):
    result = {"mentions":0,"score":50,"label":"neutre","top_posts":[],
              "bullish":0,"bearish":0,"subreddits":[]}
    bullish_words = ["buy","bull","bullish","moon","rocket","long","calls",
                     "undervalued","breakout","support","growth","upgrade",
                     "beat","strong","buy the dip","accumulate","hold"]
    bearish_words = ["sell","bear","bearish","short","puts","dump","overvalued",
                     "breakdown","resistance","decline","downgrade","miss",
                     "weak","avoid","bubble","crash"]
    subreddits  = ["wallstreetbets","stocks","investing","StockMarket"]
    headers     = {"User-Agent":"StockAI/1.0 (research bot)"}
    all_posts   = []

    for sub in subreddits:
        try:
            url    = f"https://www.reddit.com/r/{sub}/search.json"
            params = {"q":ticker,"sort":"new","limit":15,"t":"week","restrict_sr":"true"}
            r = requests.get(url, headers=headers, params=params, timeout=8)
            if r.status_code==200:
                posts = r.json().get("data",{}).get("children",[])
                for p in posts:
                    pd_data = p.get("data",{})
                    all_posts.append({
                        "title":pd_data.get("title",""),
                        "score":pd_data.get("score",0),
                        "upvote_ratio":pd_data.get("upvote_ratio",0.5),
                        "subreddit":sub,
                    })
                if posts: result["subreddits"].append(sub)
        except: continue

    if not all_posts: return result
    result["mentions"] = len(all_posts)

    bull_count=0; bear_count=0; weighted_score=0; total_weight=0
    top_posts = sorted(all_posts, key=lambda x: x["score"], reverse=True)[:5]
    result["top_posts"] = [{"title":p["title"],"score":p["score"],
                            "subreddit":p["subreddit"]} for p in top_posts]

    for post in all_posts:
        title_lower = post["title"].lower()
        post_bull = sum(1 for w in bullish_words if w in title_lower)
        post_bear = sum(1 for w in bearish_words if w in title_lower)
        bull_count += post_bull; bear_count += post_bear
        weight = max(1,post["score"])*post["upvote_ratio"]
        weighted_score += (post_bull-post_bear)*weight
        total_weight   += weight

    result["bullish"]=bull_count; result["bearish"]=bear_count
    reddit_score = normalize_score(weighted_score/total_weight if total_weight>0 else 0,-5,5)
    mention_bonus = min(10,result["mentions"]*0.5)
    if bull_count>bear_count: reddit_score=min(100,reddit_score+mention_bonus)
    elif bear_count>bull_count: reddit_score=max(0,reddit_score-mention_bonus)
    result["score"]=round(reddit_score,1)
    result["label"]=("tres_positif" if reddit_score>=70 else "positif" if reddit_score>=60
                     else "neutre" if reddit_score>=40 else "negatif"
                     if reddit_score>=30 else "tres_negatif")
    return result


# ══ GOOGLE TRENDS ════════════════════════════════════════════════════════════

def fetch_google_trends(ticker, company_name=""):
    result = {"interest_score":50,"trend_direction":"stable",
              "peak_interest":None,"current_vs_avg":None,"signal":"neutre"}
    if not TRENDS_AVAILABLE: return result
    try:
        pytrends = TrendReq(hl='fr-FR', tz=360, timeout=(10,30))
        pytrends.build_payload([ticker],cat=0,timeframe='today 3-m',geo='',gprop='')
        df_trends = pytrends.interest_over_time()
        if df_trends.empty: return result
        col    = ticker if ticker in df_trends.columns else df_trends.columns[0]
        values = df_trends[col].tolist()
        if not values: return result
        current  = float(values[-1]); avg=float(np.mean(values))
        max_val  = float(max(values))
        recent_7 = float(np.mean(values[-4:])) if len(values)>=4 else current
        older_7  = float(np.mean(values[-8:-4])) if len(values)>=8 else avg
        if avg>0:
            ratio = current/avg
            result["interest_score"] = normalize_score(ratio,0.3,3)
            result["current_vs_avg"] = round(ratio,2)
        if recent_7>older_7*1.2:
            result["trend_direction"]="croissant"; result["signal"]="interet_croissant"
        elif recent_7<older_7*0.8:
            result["trend_direction"]="decroissant"; result["signal"]="interet_decroissant"
        result["peak_interest"] = round(max_val,1)
    except Exception as e:
        print(f"Trends: {e}")
    return result


# ══ BACKTESTING ════════════════════════════════════════════════════════════

def run_backtest(ticker, rows, prices):
    if len(prices)<120:
        return {"accuracy_24h":None,"accuracy_7d":None,"backtest_score":50,
                "periods_tested":0,"note":"Donnees insuffisantes"}
    train_cutoff=len(prices)-90
    correct_24h=0; correct_7d=0; total_24h=0; total_7d=0
    mae_24h=[]; mae_7d=[]
    for i in range(train_cutoff,len(prices)-7,5):
        train_prices=prices[:i]
        if len(train_prices)<30: continue
        log_r=np.diff(np.log(train_prices)); mu=float(np.mean(log_r)); S0=train_prices[-1]
        try:
            if i+1<len(prices):
                pred=S0*np.exp(mu); actual=prices[i+1]
                if (pred>S0)==(actual>S0): correct_24h+=1
                total_24h+=1; mae_24h.append(abs(pred-actual)/actual*100)
            if i+5<len(prices):
                pred=S0*np.exp(mu*5); actual=prices[i+5]
                if (pred>S0)==(actual>S0): correct_7d+=1
                total_7d+=1; mae_7d.append(abs(pred-actual)/actual*100)
        except: continue
    acc_24h=round(correct_24h/total_24h*100,1) if total_24h>0 else None
    acc_7d =round(correct_7d/total_7d*100,1)   if total_7d>0  else None
    avg_mae=round(float(np.mean(mae_24h)),2) if mae_24h else None
    scores=[]
    if acc_24h: scores.append(normalize_score(acc_24h,40,70))
    if acc_7d:  scores.append(normalize_score(acc_7d,40,70))
    if avg_mae: scores.append(normalize_score(avg_mae,0,10,inverse=True))
    return {"accuracy_24h":acc_24h,"accuracy_7d":acc_7d,"mae_24h_pct":avg_mae,
            "backtest_score":round(sum(scores)/len(scores),1) if scores else 50,
            "periods_tested":total_24h,
            "note":f"Sur {total_24h} periodes : direction correcte {acc_24h}% a 24h, {acc_7d}% a 7j"}


# ══ DONNÉES MULTI-TIMEFRAME ═══════════════════════════════════════════════════

def safe_history(tk, period, interval, max_retries=2):
    for _ in range(max_retries):
        try:
            hist=tk.history(period=period,interval=interval)
            if not hist.empty: return hist
        except: time.sleep(1)
    return None


def fetch_data(symbol):
    tk=yf.Ticker(symbol)
    hist_1d=safe_history(tk,"5y","1d")
    if hist_1d is None or hist_1d.empty:
        hist_1d=safe_history(tk,"2y","1d")
        if hist_1d is None or hist_1d.empty:
            raise ValueError(f"Aucune donnee pour '{symbol}'")

    rows=[{"date":str(d.date()),"open":round(float(r["Open"]),4),
           "high":round(float(r["High"]),4),"low":round(float(r["Low"]),4),
           "close":round(float(r["Close"]),4),"volume":int(r["Volume"])}
          for d,r in hist_1d.iterrows()]

    rows_1h=[]; rows_5m=[]; rows_30m=[]
    for period,interval,target in [("2y","1h","1h"),("60d","5m","5m"),("60d","30m","30m")]:
        try:
            h=safe_history(tk,period,interval)
            if h is not None:
                data=[{"datetime":str(d),"open":round(float(r["Open"]),4),
                       "high":round(float(r["High"]),4),"low":round(float(r["Low"]),4),
                       "close":round(float(r["Close"]),4),"volume":int(r["Volume"])}
                      for d,r in h.iterrows()]
                if target=="1h": rows_1h=data
                elif target=="5m": rows_5m=data
                elif target=="30m": rows_30m=data
        except: pass

    info={}
    try:
        raw=tk.info
        info={
            "shortName":raw.get("shortName",symbol),"longName":raw.get("longName",symbol),
            "sector":raw.get("sector","--"),"industry":raw.get("industry","--"),
            "country":raw.get("country","--"),
            "currentPrice":raw.get("currentPrice") or rows[-1]["close"],
            "previousClose":raw.get("previousClose") or rows[-2]["close"],
            "dayChange":raw.get("regularMarketChangePercent",0),
            "currency":raw.get("currency","USD"),
            "peRatio":raw.get("trailingPE"),"forwardPE":raw.get("forwardPE"),
            "eps":raw.get("trailingEps"),"epsForward":raw.get("forwardEps"),
            "beta":raw.get("beta"),"52WeekHigh":raw.get("fiftyTwoWeekHigh"),
            "52WeekLow":raw.get("fiftyTwoWeekLow"),"marketCap":raw.get("marketCap"),
            "divYield":raw.get("dividendYield"),"roe":raw.get("returnOnEquity"),
            "roa":raw.get("returnOnAssets"),"netMargin":raw.get("profitMargins"),
            "grossMargin":raw.get("grossMargins"),"revenueGrowth":raw.get("revenueGrowth"),
            "earningsGrowth":raw.get("earningsGrowth"),"debtToEquity":raw.get("debtToEquity"),
            "freeCashflow":raw.get("freeCashflow"),"shortPct":raw.get("shortPercentOfFloat"),
            "shortRatio":raw.get("shortRatio"),"avgVolume":raw.get("averageVolume"),
            "avgVolume10d":raw.get("averageVolume10days"),
            "targetMeanPrice":raw.get("targetMeanPrice"),"targetHighPrice":raw.get("targetHighPrice"),
            "targetLowPrice":raw.get("targetLowPrice"),
            "recommendationKey":raw.get("recommendationKey"),
            "numberOfAnalysts":raw.get("numberOfAnalystOpinions"),
            "dayVolume":raw.get("regularMarketVolume"),
            "dayHigh":raw.get("regularMarketDayHigh"),"dayLow":raw.get("regularMarketDayLow"),
        }
    except:
        info={"shortName":symbol,"currency":"USD","currentPrice":rows[-1]["close"],
              "previousClose":rows[-2]["close"] if len(rows)>1 else rows[-1]["close"],"dayChange":0}
    return rows,rows_1h,rows_5m,rows_30m,info


# ══ VOLUMES TEMPS RÉEL ════════════════════════════════════════════════════════

def fetch_realtime_volume(symbol, info):
    vd={"current_volume":None,"avg_volume_10d":info.get("avgVolume10d"),
        "volume_ratio":None,"volume_signal":"neutre","buy_pressure":50,
        "sell_pressure":50,"volume_score":50,"large_trades":False,"volume_trend":"neutre"}
    try:
        tk=yf.Ticker(symbol)
        hist_1m=tk.history(period="1d",interval="1m")
        if not hist_1m.empty:
            total_vol=int(hist_1m["Volume"].sum()); vd["current_volume"]=total_vol
            avg_vol=info.get("avgVolume10d") or info.get("avgVolume")
            if avg_vol and avg_vol>0:
                hours=min(6.5,max(0.5,(datetime.utcnow().hour-13.5)))
                ratio=total_vol/max(avg_vol*(hours/6.5),1)
                vd["volume_ratio"]=round(ratio,2)
                vd["volume_signal"]=("tres_fort" if ratio>3 else "fort" if ratio>2
                                     else "eleve" if ratio>1.5 else "normal" if ratio>0.8 else "faible")
                vd["large_trades"]=ratio>3
            bull=hist_1m[hist_1m["Close"]>=hist_1m["Open"]]["Volume"].sum()
            bear=hist_1m[hist_1m["Close"]<hist_1m["Open"]]["Volume"].sum()
            tot=bull+bear
            if tot>0:
                bp=bull/tot*100
                vd["buy_pressure"]=round(float(bp),1)
                vd["sell_pressure"]=round(100-float(bp),1)
                vd["volume_score"]=round(normalize_score(vd.get("volume_ratio",1),0.3,3)*0.4+bp*0.6,1)
        hist_5d=tk.history(period="5d",interval="1d")
        if len(hist_5d)>=3:
            vols=hist_5d["Volume"].tolist()
            vd["volume_trend"]=("croissant" if vols[-1]>vols[0]*1.2
                                else "decroissant" if vols[-1]<vols[0]*0.8 else "stable")
    except Exception as e:
        print(f"Volume: {e}")
    return vd


# ══ CONTEXTE MACRO PARALLÉLISÉ ════════════════════════════════════════════════

def _fetch_single_macro(args):
    ticker_sym,period,name=args
    try:
        hist=yf.Ticker(ticker_sym).history(period=period)
        return name,(None if hist.empty else hist["Close"].tolist())
    except: return name,None


def fetch_market_context():
    context={}
    macro_list=[("^VIX","10d","vix"),("^GSPC","30d","sp500"),("^IXIC","10d","nasdaq"),
                ("GC=F","10d","gold"),("^TNX","10d","rate"),("DX-Y.NYB","10d","dollar"),
                ("^RUT","10d","russell"),("HYG","10d","hyg")]
    results={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures={ex.submit(_fetch_single_macro,t):t[2] for t in macro_list}
        for f in as_completed(futures,timeout=20):
            name,data=f.result(); results[name]=data

    if results.get("vix"):
        vd=results["vix"]; vix=float(vd[-1]); vix_5d=float(vd[-5]) if len(vd)>=5 else vix
        context.update({"vix":round(vix,2),"vix_score":compute_vix_score(vix),
                        "vix_trend":("montant" if vix>vix_5d*1.05 else "descendant" if vix<vix_5d*0.95 else "stable"),
                        "market_mood":("panique" if vix>40 else "forte_peur" if vix>30 else
                                       "peur" if vix>25 else "nerveux" if vix>20 else
                                       "neutre" if vix>15 else "euphorique")})
    if results.get("sp500"):
        sp=results["sp500"]; sp_now=float(sp[-1])
        sp_5d=float(sp[-5]) if len(sp)>=5 else sp_now
        sp_20d=float(sp[-20]) if len(sp)>=20 else sp_now
        context.update({"sp500_price":round(sp_now,2),
                        "sp500_5d_pct":round((sp_now-sp_5d)/sp_5d*100,2),
                        "sp500_20d_pct":round((sp_now-sp_20d)/sp_20d*100,2)})
        context["sp500_score"]=normalize_score(context["sp500_5d_pct"],-5,5)
        context["sp500_trend"]=("forte_hausse" if context["sp500_5d_pct"]>3 else
                                "hausse" if context["sp500_5d_pct"]>1 else
                                "stable" if context["sp500_5d_pct"]>-1 else
                                "baisse" if context["sp500_5d_pct"]>-3 else "forte_baisse")
    if results.get("nasdaq"):
        nq=results["nasdaq"]; nq_now=float(nq[-1])
        nq_5d=float(nq[-5]) if len(nq)>=5 else nq_now
        context["nasdaq_5d_pct"]=round((nq_now-nq_5d)/nq_5d*100,2)
        context["nasdaq_score"]=normalize_score(context["nasdaq_5d_pct"],-7,7)
    if results.get("gold"):
        g=results["gold"]; g_now=float(g[-1]); g_5d=float(g[-5]) if len(g)>=5 else g_now
        context["gold_5d_pct"]=round((g_now-g_5d)/g_5d*100,2)
        context["gold_score"]=normalize_score(context["gold_5d_pct"],-3,3,inverse=True)
        context["gold_signal"]="fuite_vers_securite" if context["gold_5d_pct"]>2 else "normal"
    if results.get("rate"):
        rate=float(results["rate"][-1])
        context["rate_10y"]=round(rate,3)
        context["rate_10y_score"]=normalize_score(rate,1,5,inverse=True)
    if results.get("dollar"):
        d=results["dollar"]; d_now=float(d[-1]); d_5d=float(d[-5]) if len(d)>=5 else d_now
        context["dollar_5d_pct"]=round((d_now-d_5d)/d_5d*100,2)
        context["dollar_score"]=normalize_score(context["dollar_5d_pct"],-2,2,inverse=True)
    if results.get("russell"):
        r=results["russell"]; r_now=float(r[-1]); r_5d=float(r[-5]) if len(r)>=5 else r_now
        context["russell_5d_pct"]=round((r_now-r_5d)/r_5d*100,2)
        context["russell_score"]=normalize_score(context["russell_5d_pct"],-5,5)
    if results.get("hyg"):
        h=results["hyg"]; h_now=float(h[-1]); h_5d=float(h[-5]) if len(h)>=5 else h_now
        context["hyg_5d_pct"]=round((h_now-h_5d)/h_5d*100,2)
        context["credit_signal"]="risk_on" if context["hyg_5d_pct"]>0 else "risk_off"

    context["macro_score"]=compute_macro_score(
        context.get("sp500_5d_pct"),context.get("nasdaq_5d_pct"),
        context.get("gold_5d_pct"),context.get("dollar_5d_pct"),context.get("rate_10y"))
    return context


# ══ NEWS + SENTIMENT ═══════════════════════════════════════════════════════════

def fetch_news(symbol):
    news=[]
    if not FINNHUB_KEY: return news
    try:
        today=(date.today()).strftime("%Y-%m-%d")
        last_week=(date.today()-timedelta(days=14)).strftime("%Y-%m-%d")
        r=requests.get(f"https://finnhub.io/api/v1/company-news"
                       f"?symbol={symbol}&from={last_week}&to={today}&token={FINNHUB_KEY}",timeout=8)
        news=[{"headline":a.get("headline",""),"summary":a.get("summary","")[:200],
               "source":a.get("source",""),"datetime":a.get("datetime",0)}
              for a in r.json()[:12] if a.get("headline")]
    except: pass
    return news


def analyze_sentiment(symbol, news, market_context):
    if not news or not GROQ_KEY:
        return {"score":50,"raw_score":0,"label":"neutre","resume":"Pas de news",
                "details":[],"macro_impact":"neutre","news_count":0}
    headlines=[n["headline"] for n in news[:10]]
    vix=market_context.get("vix",20); sp5=market_context.get("sp500_5d_pct",0)
    mood=market_context.get("market_mood","neutre")
    prompt=(f"Analyse sentiment pour {symbol}.\n"
            f"NEWS: {chr(10).join(f'- {h}' for h in headlines)}\n"
            f"MACRO: VIX:{vix} ({mood}) | SP500:{sp5:+.1f}%\n"
            f'Reponds UNIQUEMENT JSON: {{"raw_score":<-100 a 100>,"label":"<tres_negatif|negatif|neutre|positif|tres_positif>","resume":"<1 phrase>","macro_impact":"<negatif|neutre|positif>","details":["<p1>","<p2>","<p3>"]}}')
    try:
        r=requests.post("https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"},
                        json={"model":"llama-3.3-70b-versatile",
                              "messages":[{"role":"user","content":prompt}],
                              "max_tokens":200,"temperature":0.1},timeout=20)
        content=r.json()["choices"][0]["message"]["content"].strip()
        start=content.find("{"); end=content.rfind("}")+1
        result=json.loads(content[start:end])
        raw=result.get("raw_score",0)
        result["score"]=round((raw+100)/2,1); result["raw_score"]=raw
        result["news_count"]=len(news)
        return result
    except:
        return {"score":50,"raw_score":0,"label":"neutre","resume":"Indisponible",
                "details":[],"macro_impact":"neutre","news_count":len(news)}


def apply_sentiment(predictions, sentiment_score_100, market_context):
    sent_adj=(sentiment_score_100-50)*0.25
    vix=market_context.get("vix",20); sp_trend=market_context.get("sp500_5d_pct",0)
    vix_adj=-min(10,max(0,(vix-20)*0.3)) if vix>20 else 0
    total=max(-25,min(25,sent_adj+vix_adj+sp_trend*0.5))
    adjusted={}
    for mk,horizons in predictions.items():
        adjusted[mk]={}
        for h,pred in horizons.items():
            p=pred.copy()
            p["prob_up"]=round(min(95,max(5,p["prob_up"]+total)),1)
            p["sentiment_adjustment"]=round(total,1)
            adjusted[mk][h]=p
    return adjusted


# ══ FEATURES TECHNIQUES ═══════════════════════════════════════════════════════

def build_features(rows, is_intraday=False):
    if not ML_AVAILABLE or not rows: return None
    try:
        df=pd.DataFrame(rows)
        if is_intraday and "datetime" in df.columns:
            df["date"]=pd.to_datetime(df["datetime"])
        else:
            df["date"]=pd.to_datetime(df["date"])
        df.set_index("date",inplace=True); df.sort_index(inplace=True)
        c=df["close"]
        has_ohlv=all(col in df.columns for col in ["open","high","low","volume"])

        df["ma5"]=c.rolling(5).mean(); df["ma10"]=c.rolling(10).mean()
        df["ma20"]=c.rolling(20).mean()
        df["ma50"]=c.rolling(min(50,max(2,len(df)//4))).mean()
        df["ma200"]=c.rolling(min(200,len(df)//2)).mean() if len(df)>=40 else df["ma50"]
        df["above_ma200"]=(c>df["ma200"]).astype(int)
        df["above_ma50"]=(c>df["ma50"]).astype(int)
        df["golden_cross"]=((df["ma50"]>df["ma200"]) & (df["ma50"].shift(1)<=df["ma200"].shift(1))).astype(int)
        df["ema12"]=c.ewm(span=12).mean(); df["ema26"]=c.ewm(span=26).mean()

        delta=c.diff()
        gain=delta.clip(lower=0).rolling(14).mean()
        loss=(-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"]=100-100/(1+gain/loss.replace(0,np.nan))
        gain2=delta.clip(lower=0).rolling(2).mean()
        loss2=(-delta.clip(upper=0)).rolling(2).mean()
        df["rsi2"]=100-100/(1+gain2/loss2.replace(0,np.nan))

        df["macd"]=df["ema12"]-df["ema26"]
        df["macd_signal"]=df["macd"].ewm(span=9).mean()
        df["macd_hist"]=df["macd"]-df["macd_signal"]
        df["macd_cross"]=((df["macd"]>df["macd_signal"]) & (df["macd"].shift(1)<=df["macd_signal"].shift(1))).astype(int)

        bb_mid=c.rolling(20).mean(); bb_std=c.rolling(20).std()
        df["bb_pct"]=(c-bb_mid)/(2*bb_std+1e-9)
        df["bb_width"]=(bb_mid+2*bb_std-(bb_mid-2*bb_std))/(bb_mid+1e-9)
        df["bb_squeeze"]=(df["bb_width"]<df["bb_width"].rolling(20).mean()*0.85).astype(int)

        if has_ohlv:
            h=df["high"]; l=df["low"]; v=df["volume"]
            tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
            df["atr"]=tr.rolling(14).mean(); df["atr_pct"]=df["atr"]/(c+1e-9)
            low14=l.rolling(14).min(); high14=h.rolling(14).max()
            df["stoch_k"]=100*(c-low14)/(high14-low14+1e-9)
            df["stoch_d"]=df["stoch_k"].rolling(3).mean()
            df["williams_r"]=-100*(high14-c)/(high14-low14+1e-9)
            tp=(h+l+c)/3
            df["cci"]=(tp-tp.rolling(20).mean())/(0.015*tp.rolling(20).std()+1e-9)
            df["vol_ratio"]=v/(v.rolling(20).mean()+1)
            df["vol_spike"]=(df["vol_ratio"]>2).astype(int)
            df["vwap"]=(tp*v).cumsum()/(v.cumsum()+1e-9)
            df["above_vwap"]=(c>df["vwap"]).astype(int)
            obv=[0]
            for i in range(1,len(df)):
                if df["close"].iloc[i]>df["close"].iloc[i-1]: obv.append(obv[-1]+v.iloc[i])
                elif df["close"].iloc[i]<df["close"].iloc[i-1]: obv.append(obv[-1]-v.iloc[i])
                else: obv.append(obv[-1])
            df["obv_trend"]=pd.Series(obv,index=df.index).rolling(10).mean()
        else:
            df["atr_pct"]=c.rolling(14).std()/(c+1e-9)
            df["stoch_k"]=50.0; df["stoch_d"]=50.0; df["williams_r"]=-50.0
            df["cci"]=0.0; df["vol_ratio"]=1.0; df["vol_spike"]=0
            df["vwap"]=c; df["above_vwap"]=0; df["obv_trend"]=0.0

        df["log_ret"]=np.log(c/(c.shift(1)+1e-9))
        df["ret_1d"]=c.pct_change(1); df["ret_5d"]=c.pct_change(5); df["ret_20d"]=c.pct_change(20)
        df["volatility"]=df["log_ret"].rolling(20).std()*np.sqrt(252)
        df["mom_10"]=c/(c.shift(10)+1e-9)-1; df["mom_20"]=c/(c.shift(20)+1e-9)-1
        df["dist_52w_high"]=c/(c.rolling(min(252,len(c))).max()+1e-9)-1
        df["dist_52w_low"]=c/(c.rolling(min(252,len(c))).min()+1e-9)-1
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Features: {e}"); return None


def compute_technical_signals(df):
    if df is None or len(df)<2:
        return {"score":50,"signals":{},"label":"neutre","rsi":50,"macd":0,"atr_pct":0,"bb_pct":0,"stoch":50}
    last=df.iloc[-1]; signals={}

    if last.get("above_ma200",0): signals["Tendance long terme"]=("HAUSSIER",+15)
    else: signals["Tendance long terme"]=("BAISSIER",-15)
    if last.get("above_ma50",0): signals["Tendance moyen terme"]=("HAUSSIER",+10)
    else: signals["Tendance moyen terme"]=("BAISSIER",-10)

    ma5,ma20=last.get("ma5",0),last.get("ma20",0)
    signals["MA5/MA20"]=("HAUSSIER",+8) if ma5>ma20 else ("BAISSIER",-8)

    rsi=last.get("rsi",50)
    if rsi>70: signals["RSI"]=("SURACHETE",-12)
    elif rsi<30: signals["RSI"]=("SURVENDU",+12)
    elif rsi>55: signals["RSI"]=("HAUSSIER",+6)
    elif rsi<45: signals["RSI"]=("BAISSIER",-6)
    else: signals["RSI"]=("NEUTRE",0)

    signals["MACD"]=("HAUSSIER",+10) if last.get("macd_hist",0)>0 else ("BAISSIER",-10)
    if last.get("macd_cross",0): signals["Signal MACD"]=("CROISEMENT",+5)

    bb=last.get("bb_pct",0)
    if bb>0.8: signals["Bollinger"]=("SURACHETE",-8)
    elif bb<-0.8: signals["Bollinger"]=("SURVENDU",+8)
    elif bb>0.2: signals["Bollinger"]=("HAUSSIER",+4)
    else: signals["Bollinger"]=("BAISSIER",-4)

    stoch=last.get("stoch_k",50)
    if stoch>80: signals["Stochastique"]=("SURACHETE",-6)
    elif stoch<20: signals["Stochastique"]=("SURVENDU",+6)
    else: signals["Stochastique"]=("NEUTRE",0)

    if last.get("vol_spike",0): signals["Volume Spike"]=("VOLUME FORT",+5)
    if last.get("above_vwap",0): signals["VWAP"]=("AU-DESSUS VWAP",+6)
    else: signals["VWAP"]=("SOUS VWAP",-6)

    wr=last.get("williams_r",-50)
    if wr>-20: signals["Williams %R"]=("SURACHETE",-5)
    elif wr<-80: signals["Williams %R"]=("SURVENDU",+5)
    else: signals["Williams %R"]=("NEUTRE",0)

    if last.get("golden_cross",0): signals["Golden Cross"]=("SIGNAL FORT",+20)
    if last.get("bb_squeeze",0): signals["BB Squeeze"]=("EXPLOSION IMMINENTE",+3)

    mom20=last.get("mom_20",0)
    if mom20>0.05: signals["Momentum 20j"]=("FORT",+8)
    elif mom20>0: signals["Momentum 20j"]=("POSITIF",+4)
    elif mom20>-0.05: signals["Momentum 20j"]=("FAIBLE",-4)
    else: signals["Momentum 20j"]=("NEGATIF",-8)

    raw_score=max(-100,min(100,sum(v[1] for v in signals.values())))
    score_100=round((raw_score+100)/2,1)
    label=("Fortement haussier" if raw_score>60 else "Haussier" if raw_score>25 else
           "Legt. haussier" if raw_score>5 else "Neutre" if raw_score>-5 else
           "Legt. baissier" if raw_score>-25 else "Baissier" if raw_score>-60 else "Fortement baissier")
    return {"score":score_100,"raw_score":raw_score,"label":label,
            "signals":{k:{"signal":v[0],"score":v[1]} for k,v in signals.items()},
            "rsi":round(float(rsi),1),"macd":round(float(last.get("macd",0)),4),
            "atr_pct":round(float(last.get("atr_pct",0))*100,2),
            "bb_pct":round(float(last.get("bb_pct",0)),2),"stoch":round(float(stoch),1)}


# ══ FINNHUB ════════════════════════════════════════════════════════════════════

def fetch_finnhub_all(symbol):
    result={"recs":[],"earnings":[],"peers":[],"metrics":{}}
    if not FINNHUB_KEY: return result
    base="https://finnhub.io/api/v1"; hdrs={"X-Finnhub-Token":FINNHUB_KEY}
    for endpoint,key in [
        (f"{base}/stock/recommendation?symbol={symbol}","recs"),
        (f"{base}/stock/earnings?symbol={symbol}","earnings"),
        (f"{base}/stock/peers?symbol={symbol}","peers"),
    ]:
        try:
            r=requests.get(endpoint,headers=hdrs,timeout=6); data=r.json()
            result[key]=(data[:3] if key=="recs" else data[:4] if key=="earnings" else data[:5])
        except: pass
    try:
        r=requests.get(f"{base}/stock/metric?symbol={symbol}&metric=all",headers=hdrs,timeout=6)
        m=r.json().get("metric",{})
        result["metrics"]={"beta5Y":m.get("beta5Y"),"roeTTM":m.get("roeTTM"),
                           "roiTTM":m.get("roiTTM"),"netMarginTTM":m.get("netProfitMarginTTM"),
                           "peExclExtraTTM":m.get("peExclExtraTTM"),"pbAnnual":m.get("pbAnnual")}
    except: pass
    return result


# ══ MONTE CARLO ════════════════════════════════════════════════════════════════

def monte_carlo(prices, horizons, n=8000):
    log_r=np.diff(np.log(prices)); mu=float(np.mean(log_r)); sigma=float(np.std(log_r)); S0=prices[-1]
    mu_adj=mu+(prices[-1]-prices[-20])/prices[-20]*0.01 if len(prices)>=20 else mu
    out={}
    for label,T in horizons.items():
        steps=max(1,int(T*252)); dt=T/steps
        Z=np.random.standard_normal((n,steps))
        paths=S0*np.exp(np.cumsum((mu_adj-.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z,axis=1))
        final=paths[:,-1]; mean_p=float(np.mean(final)); std_p=float(np.std(final))
        pct=(mean_p-S0)/S0*100; prob_u=float(np.mean(final>S0)*100)
        qual=max(0,min(100,int(100*(1-min(std_p/(mean_p+1e-9)*2,1)))))
        out[label]={"model":"Monte Carlo","price":round(mean_p,4),"pct_change":round(pct,2),
                    "prob_up":round(prob_u,1),"ci_low":round(float(np.percentile(final,5)),4),
                    "ci_high":round(float(np.percentile(final,95)),4),"quality":qual}
    return out


# ══ DIVERSIFICATION HORIZONS ════════════════════════════════════════════════

def diversify_predictions(predictions, prices, info):
    S0=prices[-1]; target=info.get("targetMeanPrice")
    trend_5d=(prices[-1]/prices[-5]-1) if len(prices)>=5 else 0
    trend_20d=(prices[-1]/prices[-20]-1) if len(prices)>=20 else 0
    trend_60d=(prices[-1]/prices[-60]-1) if len(prices)>=60 else 0
    horizon_days={"1h":0.04,"6h":0.25,"24h":1,"7d":5,"1m":21,"6m":126}

    for model_key in ["xgboost","ensemble"]:
        model_preds=predictions.get(model_key,{})
        if not model_preds: continue
        values=[v.get("price",0) for v in model_preds.values()]
        if len(set(round(v,2) for v in values))>2: continue
        ref_pred=model_preds.get("24h") or list(model_preds.values())[0]
        if not ref_pred: continue
        ref_pct=ref_pred.get("pct_change",0)
        for label,days in horizon_days.items():
            if label not in predictions.get(model_key,{}): continue
            if days<=1: new_pct=ref_pct
            elif days<=5: new_pct=ref_pct*0.7+trend_5d*100*0.3
            elif days<=21:
                ap=(target/S0-1)*0.15*100 if target and S0>0 else 0
                new_pct=ref_pct*0.4+trend_20d*100*0.3+ap*0.3
            else:
                ap=(target/S0-1)*0.35*100 if target and S0>0 else 0
                new_pct=ref_pct*0.2+trend_60d*100*0.2+ap*0.6
            new_price=round(S0*(1+new_pct/100),4)
            old=predictions[model_key][label]
            predictions[model_key][label]={**old,"price":new_price,
                "pct_change":round(new_pct,2),"prob_up":round(min(95,max(5,50+new_pct*3)),1),
                "diversified":True}

    mc_r=predictions.get("monte_carlo",{}); xgb_r=predictions.get("xgboost",{})
    lstm_r=predictions.get("lstm",{})
    W=({"Monte Carlo":.20,"XGBoost":.45,"LSTM":.35} if LSTM_AVAILABLE and lstm_r
       else {"Monte Carlo":.35,"XGBoost":.65})
    for h in set(mc_r)|set(xgb_r)|set(lstm_r):
        parts=[d[h] for d in [mc_r,xgb_r,lstm_r] if h in d]
        if not parts: continue
        tw=sum(W.get(p["model"],.33) for p in parts)
        def wa(k): return sum(p[k]*W.get(p["model"],.33) for p in parts)/tw
        predictions["ensemble"][h]={**predictions["ensemble"].get(h,{}),
            "price":round(wa("price"),4),"pct_change":round(wa("pct_change"),2),
            "prob_up":round(min(95,max(5,wa("prob_up"))),1),
            "quality":int(wa("quality")) if parts and "quality" in parts[0] else 50,
            "models_used":[p["model"] for p in parts]}
    return predictions


# ══ XGBOOST ════════════════════════════════════════════════════════════════════

FEAT=["ma5","ma10","ma20","rsi","rsi2","macd","macd_hist","bb_pct","bb_width",
      "stoch_k","stoch_d","atr_pct","cci","vol_ratio","volatility",
      "mom_10","mom_20","ret_5d","dist_52w_high","above_ma200","williams_r","above_vwap"]


def train_xgb_multitf(ticker,rows_1d,rows_1h,rows_5m,rows_30m):
    results={}; models={}; feat_map={}
    config={"1h":(rows_5m,12,True),"6h":(rows_30m,12,True),"24h":(rows_1h,24,True),
            "7d":(rows_1d,7,False),"1m":(rows_1d,21,False),"6m":(rows_1d,126,False)}
    for label,(rows,shift,is_intraday) in config.items():
        if not rows or len(rows)<max(30,shift*3): continue
        try:
            df=build_features(rows,is_intraday=is_intraday)
            if df is None or len(df)<shift*2: continue
            S0=float(df["close"].iloc[-1])
            fc=[f for f in FEAT if f in df.columns]
            d2=df.copy(); d2["target"]=d2["close"].shift(-shift); d2.dropna(inplace=True)
            X,y=d2[fc].values,d2["target"].values
            split=max(15,int(len(X)*0.8))
            if split>=len(X)-1: continue
            mdl=xgb.XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.04,
                                  subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
            es=[(X[split:],y[split:])] if len(X[split:])>0 else None
            mdl.fit(X[:split],y[:split],eval_set=es,verbose=False)
            pred=float(mdl.predict(X[-1].reshape(1,-1))[0]); pct=(pred-S0)/S0*100
            qual=60
            if len(X[split:]):
                rmse=float(np.sqrt(mean_squared_error(y[split:],mdl.predict(X[split:]))))
                qual=max(0,min(100,int(100*(1-min(rmse/(S0+1e-9)*10,1)))))
            models[label]=mdl; feat_map[label]=fc
            tf=("5min" if label=="1h" else "30min" if label=="6h" else "1h" if label=="24h" else "1j")
            results[label]={"model":"XGBoost","price":round(pred,4),"pct_change":round(pct,2),
                            "prob_up":round(min(95,max(5,50+pct*3)),1),
                            "quality":qual,"timeframe":tf,"data_points":len(df)}
        except Exception as e:
            print(f"XGB {label}: {e}"); continue
    if ticker not in MODEL_CACHE: MODEL_CACHE[ticker]={}
    MODEL_CACHE[ticker].update({"xgb_models":models,"xgb_results":results,
                                "xgb_feat":feat_map,"trained_at":time.time()})
    return results


def pred_xgb_multitf(ticker,rows_1d,rows_1h,rows_5m,rows_30m):
    cache=MODEL_CACHE.get(ticker,{})
    if cache.get("xgb_models") and cache_valid(ticker):
        results={}
        row_map={"1h":rows_5m,"6h":rows_30m,"24h":rows_1h,"7d":rows_1d,"1m":rows_1d,"6m":rows_1d}
        for label,mdl in cache["xgb_models"].items():
            try:
                rows=row_map.get(label,[]); is_i=label in ["1h","6h","24h"]
                df=build_features(rows,is_intraday=is_i)
                if df is None or len(df)<5: continue
                S0=float(df["close"].iloc[-1])
                fc=[f for f in cache["xgb_feat"].get(label,FEAT) if f in df.columns]
                pred=float(mdl.predict(df[fc].iloc[-1].values.reshape(1,-1))[0])
                pct=(pred-S0)/S0*100; old=cache["xgb_results"].get(label,{})
                results[label]={"model":"XGBoost","price":round(pred,4),"pct_change":round(pct,2),
                                "prob_up":round(min(95,max(5,50+pct*3)),1),
                                "quality":old.get("quality",60),"from_cache":True,
                                "timeframe":old.get("timeframe","1j")}
            except Exception as e:
                print(f"Cache XGB {label}: {e}")
        if results: return results
    return train_xgb_multitf(ticker,rows_1d,rows_1h,rows_5m,rows_30m)


# ══ LSTM ═══════════════════════════════════════════════════════════════════════

def train_lstm(ticker,prices):
    if not LSTM_AVAILABLE or len(prices)<50: return {}
    try:
        SEQ=30; sc=MinMaxScaler()
        scp=sc.fit_transform(np.array(prices).reshape(-1,1))
        X,y=[],[]
        for i in range(SEQ,len(scp)):
            X.append(scp[i-SEQ:i,0]); y.append(scp[i,0])
        X=np.array(X).reshape(-1,SEQ,1); y=np.array(y)
        sp=max(15,int(len(X)*0.85))
        mdl=Sequential([LSTM(128,return_sequences=True,input_shape=(SEQ,1)),Dropout(0.2),
                        LSTM(64,return_sequences=True),Dropout(0.2),
                        LSTM(32),Dropout(0.2),Dense(16,activation="relu"),Dense(1)])
        mdl.compile(optimizer="adam",loss="huber")
        es=EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True,verbose=0)
        mdl.fit(X[:sp],y[:sp],epochs=60,batch_size=16,verbose=0,
                validation_split=0.1,callbacks=[es])
        S0=prices[-1]
        pred=float(sc.inverse_transform([[float(mdl.predict(scp[-SEQ:].reshape(1,SEQ,1),verbose=0)[0][0])]])[0][0])
        pct=(pred-S0)/S0*100; qual=58
        if len(X)>sp:
            pt=sc.inverse_transform(mdl.predict(X[sp:],verbose=0))
            at=sc.inverse_transform(y[sp:].reshape(-1,1))
            rmse=float(np.sqrt(mean_squared_error(at,pt)))
            qual=max(0,min(100,int(100*(1-min(rmse/(S0+1e-9)*10,1)))))
        if ticker not in MODEL_CACHE: MODEL_CACHE[ticker]={}
        MODEL_CACHE[ticker].update({"lstm_model":mdl,"lstm_scaler":sc,"lstm_qual":qual,"lstm_SEQ":SEQ})
        return {lbl:{"model":"LSTM","price":round(pred,4),"pct_change":round(pct,2),
                     "prob_up":round(min(95,max(5,50+pct*4)),1),"quality":qual}
                for lbl in ["1h","6h","24h","7d","1m","6m"]}
    except Exception as e:
        print(f"LSTM: {e}"); return {}


def pred_lstm(ticker,prices):
    cache=MODEL_CACHE.get(ticker,{})
    if cache.get("lstm_model") and cache_valid(ticker):
        try:
            sc,mdl=cache["lstm_scaler"],cache["lstm_model"]
            SEQ,qual=cache["lstm_SEQ"],cache["lstm_qual"]; S0=prices[-1]
            scp=sc.transform(np.array(prices).reshape(-1,1))
            pred=float(sc.inverse_transform([[float(mdl.predict(scp[-SEQ:].reshape(1,SEQ,1),verbose=0)[0][0])]])[0][0])
            pct=(pred-S0)/S0*100
            return {lbl:{"model":"LSTM","price":round(pred,4),"pct_change":round(pct,2),
                         "prob_up":round(min(95,max(5,50+pct*4)),1),"quality":qual,"from_cache":True}
                    for lbl in ["1h","6h","24h","7d","1m","6m"]}
        except: pass
    return train_lstm(ticker,prices)


def make_ensemble(mc,xgb_r,lstm_r,tech_score_100=50,volume_score=50,options_score=50):
    W=({"Monte Carlo":.18,"XGBoost":.45,"LSTM":.37} if LSTM_AVAILABLE and lstm_r
       else {"Monte Carlo":.35,"XGBoost":.65})
    out={}
    for h in set(mc)|set(xgb_r)|set(lstm_r):
        parts=[d[h] for d in [mc,xgb_r,lstm_r] if h in d]
        if not parts: continue
        tw=sum(W.get(p["model"],.33) for p in parts)
        def wa(k): return sum(p[k]*W.get(p["model"],.33) for p in parts)/tw
        tech_adj=(tech_score_100-50)*0.08
        vol_adj=(volume_score-50)*0.04
        opt_adj=(options_score-50)*0.04
        out[h]={"model":"Ensemble","price":round(wa("price"),4),
                "pct_change":round(wa("pct_change"),2),
                "prob_up":round(min(95,max(5,wa("prob_up")+tech_adj+vol_adj+opt_adj)),1),
                "quality":int(wa("quality")),"models_used":[p["model"] for p in parts]}
    return out


# ══ ANALYSE IA GOLDMAN SACHS ════════════════════════════════════════════════

def groq_masterclass_gs(symbol, info, preds, sentiment, tech_signals,
                         market_ctx, fh_data, rows, volume_data,
                         reddit_data, trends_data, backtest,
                         options_data, insider_data, short_data,
                         sector_momentum, earnings_quality):
    if not GROQ_KEY:
        return _fallback(symbol,info,preds,sentiment,tech_signals)

    ens=preds.get("ensemble",{})
    h1h=ens.get("1h",{}); h6h=ens.get("6h",{}); h24=ens.get("24h",{})
    h7d=ens.get("7d",{}); h1m=ens.get("1m",{}); h6m=ens.get("6m",{})

    recs=fh_data.get("recs",[]); peers=fh_data.get("peers",[])
    prices_list=[r["close"] for r in rows[-60:]]
    perf_1m=round((prices_list[-1]/prices_list[-20]-1)*100,2) if len(prices_list)>=20 else None
    perf_3m=round((prices_list[-1]/prices_list[-60]-1)*100,2) if len(prices_list)>=60 else None

    cons_text="Non disponible"
    if recs:
        r0=recs[0]; tot=sum(r0.get(k,0) for k in ["strongBuy","buy","hold","sell","strongSell"])
        if tot: cons_text=(f"Fort Achat:{r0.get('strongBuy',0)} Achat:{r0.get('buy',0)} "
                          f"Neutre:{r0.get('hold',0)} Vente:{r0.get('sell',0)} ({tot} analystes)")

    current=info.get("currentPrice",0); target=info.get("targetMeanPrice")
    upside=(f"{round((target/current-1)*100,1):+.1f}%" if target and current else "N/A")

    prompt=f"""Tu es un analyste senior Goldman Sachs. Redige une analyse MASTERCLASS ultra-complete en francais pour {symbol} ({info.get('shortName','')}).
Utilise TOUTES les donnees. Sois precis, chiffre tout, analyse chaque signal.

━━━ DONNEES DE MARCHE ━━━
Prix: {current} {info.get('currency','USD')} ({info.get('dayChange',0):+.2f}%) | Perf 1m:{perf_1m}% 3m:{perf_3m}%
52W: {info.get('52WeekHigh')}/{info.get('52WeekLow')}

━━━ ANALYSTES ━━━
Objectif: {target} ({upside}) | {info.get('numberOfAnalysts','?')} analystes | Rec:{info.get('recommendationKey')}
Consensus: {cons_text}
Earnings quality: {earnings_quality.get('label','N/A')} (beat rate {earnings_quality.get('beat_rate','?')}%, surprise moy {earnings_quality.get('avg_surprise','?')}%)

━━━ FONDAMENTAUX ━━━
P/E:{info.get('peRatio')}/{info.get('forwardPE')} | ROE:{round((info.get('roe',0) or 0)*100,1)}%
Marge:{round((info.get('netMargin',0) or 0)*100,1)}% | Croissance:{round((info.get('revenueGrowth',0) or 0)*100,1)}%
FCF:{info.get('freeCashflow')} | Dette/Cap:{info.get('debtToEquity')}

━━━ OPTIONS FLOW ━━━
Put/Call ratio: {options_data.get('put_call_ratio','N/A')} | Signal: {options_data.get('signal','neutre')}
Calls volume: {options_data.get('total_call_volume','N/A')} | Puts volume: {options_data.get('total_put_volume','N/A')}
IV Calls: {options_data.get('avg_call_iv','N/A')}% | IV Puts: {options_data.get('avg_put_iv','N/A')}%
Activite inhabituelle: {options_data.get('unusual_activity',False)}

━━━ INSIDER TRADING ━━━
Signal: {insider_data.get('signal','neutre')} | Score: {insider_data.get('score',50)}/100
{insider_data.get('label','N/A')}
Achats: ${insider_data.get('buy_value',0):,.0f} | Ventes: ${insider_data.get('sell_value',0):,.0f}
Acheteurs: {insider_data.get('buyers',0)} | Vendeurs: {insider_data.get('sellers',0)}

━━━ SHORT INTEREST ━━━
Short % flottant: {short_data.get('short_pct_float','N/A')}%
Jours pour couvrir: {short_data.get('days_to_cover','N/A')}
Risque squeeze: {short_data.get('squeeze_risk','faible')}
Signal: {short_data.get('label','N/A')}

━━━ MOMENTUM RELATIF ━━━
Vs SP500 1m: {sector_momentum.get('vs_sp500_1m','N/A')}% | 3m: {sector_momentum.get('vs_sp500_3m','N/A')}%
Vs secteur 1m: {sector_momentum.get('vs_sector_1m','N/A')}%
Force relative: {sector_momentum.get('relative_strength',50)}/100 ({sector_momentum.get('signal','neutre')})

━━━ SCORES /100 ━━━
Technique:{tech_signals.get('score',50)} ({tech_signals.get('label','')})
RSI:{tech_signals.get('rsi')} | MACD:{tech_signals.get('macd')} | Stoch:{tech_signals.get('stoch')}
Sentiment news:{sentiment.get('score',50)} ({sentiment.get('label','')})
{sentiment.get('resume','')}
Options:{options_data.get('score',50)} | Insider:{insider_data.get('score',50)}
Reddit:{reddit_data.get('score',50)} ({reddit_data.get('mentions',0)} mentions bull:{reddit_data.get('bullish',0)} bear:{reddit_data.get('bearish',0)})
Google Trends:{trends_data.get('interest_score',50)} ({trends_data.get('trend_direction','stable')})
Macro:{market_ctx.get('macro_score',50)} VIX:{market_ctx.get('vix')}({market_ctx.get('vix_score',50)})
SP500:{market_ctx.get('sp500_5d_pct',0):+.1f}%({market_ctx.get('sp500_score',50)})
Volume:{volume_data.get('volume_score',50)} PressionAchat:{volume_data.get('buy_pressure',50)}%
Russell:{market_ctx.get('russell_5d_pct',0):+.1f}% | Credit HYG:{market_ctx.get('hyg_5d_pct',0):+.1f}% ({market_ctx.get('credit_signal','neutre')})
Backtest 90j: precision 24h={backtest.get('accuracy_24h','N/A')}% 7j={backtest.get('accuracy_7d','N/A')}%

━━━ PREDICTIONS ENSEMBLE ━━━
1h:{h1h.get('pct_change','?')}% prob↑{h1h.get('prob_up','?')}% q:{h1h.get('quality','?')}/100
6h:{h6h.get('pct_change','?')}% prob↑{h6h.get('prob_up','?')}%
24h:{h24.get('pct_change','?')}% prob↑{h24.get('prob_up','?')}% q:{h24.get('quality','?')}/100
7j:{h7d.get('pct_change','?')}% prob↑{h7d.get('prob_up','?')}%
1m:{h1m.get('pct_change','?')}% prob↑{h1m.get('prob_up','?')}% q:{h1m.get('quality','?')}/100
6m:{h6m.get('pct_change','?')}% prob↑{h6m.get('prob_up','?')}%
PAIRS: {', '.join(peers[:4]) if peers else 'N/A'}

FORMAT MARKDOWN OBLIGATOIRE — sois exhaustif et precis :
## Synthese Executive (verdict immediat)
## Analyse Fondamentale et Qualite des Resultats
## Analyse Technique Avancee
## Options Flow et Positionnement Institutionnel
## Insider Trading et Signal Dirigeants
## Short Interest et Risque de Squeeze
## Momentum Relatif et Force Sectorielle
## Sentiment Social et Tendances Google
## Contexte Macro (VIX, SP500, Credit, Russell)
## Fiabilite des Modeles (Backtest)
## Predictions Probabilistes par Horizon
## Catalyseurs Haussiers vs Risques Baissiers
## Verdict Final Goldman Sachs (biais, conviction 1-10, cible de prix, strategie)"""

    try:
        r=requests.post("https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"},
                        json={"model":"llama-3.3-70b-versatile",
                              "messages":[{"role":"user","content":prompt}],
                              "max_tokens":2500,"temperature":0.3},timeout=90)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq: {e}"); return _fallback(symbol,info,preds,sentiment,tech_signals)


def _fallback(symbol,info,preds,sentiment,tech_signals):
    ens=preds.get("ensemble",{}); h24=ens.get("24h",{}); h1m=ens.get("1m",{})
    chg=info.get("dayChange",0) or 0
    return (f"## Analyse {symbol}\nPrix: {info.get('currentPrice')} ({chg:+.2f}%)\n"
            f"Technique: {tech_signals.get('label')} ({tech_signals.get('score',50)}/100)\n"
            f"Sentiment: {sentiment.get('label')} ({sentiment.get('score',50)}/100)\n"
            f"24h: {h24.get('pct_change','N/A')}% | 1m: {h1m.get('pct_change','N/A')}%")


# ══ VERDICT ACHAT PAR HORIZON ════════════════════════════════════════════════

def build_horizon_recommendation(preds_adj, global_score, tech_score, fund_score, sentiment_score):
    ens = preds_adj.get("ensemble", {})

    def horizon_block(label, keys, weights):
        vals = []
        for k, w in zip(keys, weights):
            p = ens.get(k, {})
            if p and p.get("pct_change") is not None:
                vals.append((float(p.get("pct_change", 0)), float(p.get("prob_up", 50)), w))
        if not vals:
            return {
                "horizon": label,
                "verdict": "indetermine",
                "confidence": 45,
                "expected_change_pct": None,
                "prob_up": None,
                "label": "Donnees insuffisantes",
            }

        tw = sum(v[2] for v in vals)
        exp_pct = sum(v[0] * v[2] for v in vals) / tw
        prob_up = sum(v[1] * v[2] for v in vals) / tw

        base_conf = (
            global_score * 0.45 +
            tech_score * 0.20 +
            fund_score * 0.20 +
            sentiment_score * 0.15
        )
        confidence = max(5, min(95, round(base_conf + (prob_up - 50) * 0.2, 1)))

        if exp_pct >= 3 and prob_up >= 62:
            verdict = "achat_fort"
            label_txt = "Achat fort"
        elif exp_pct >= 1 and prob_up >= 55:
            verdict = "achat"
            label_txt = "Achat"
        elif exp_pct > -1 and prob_up >= 45:
            verdict = "surveiller"
            label_txt = "Surveiller / attendre"
        else:
            verdict = "eviter"
            label_txt = "Eviter pour l'instant"

        return {
            "horizon": label,
            "verdict": verdict,
            "confidence": confidence,
            "expected_change_pct": round(exp_pct, 2),
            "prob_up": round(prob_up, 1),
            "label": label_txt,
        }

    short_term = horizon_block("court_terme", ["1h", "6h", "24h"], [0.2, 0.3, 0.5])
    mid_term = horizon_block("moyen_terme", ["7d", "1m"], [0.45, 0.55])
    long_term = horizon_block("long_terme", ["6m"], [1.0])

    def traffic_light(verdict):
        if verdict in ["achat_fort", "achat"]:
            return "vert"
        if verdict == "surveiller":
            return "orange"
        if verdict == "eviter":
            return "rouge"
        return "gris"

    def short_word(verdict):
        if verdict in ["achat_fort", "achat"]:
            return "ACHAT"
        if verdict == "surveiller":
            return "ATTENDRE"
        if verdict == "eviter":
            return "EVITER"
        return "INDETERMINE"

    quick_summary = (
        f"Court terme: {short_word(short_term.get('verdict'))} | "
        f"Moyen terme: {short_word(mid_term.get('verdict'))} | "
        f"Long terme: {short_word(long_term.get('verdict'))}"
    )

    return {
        "court_terme": short_term,
        "moyen_terme": mid_term,
        "long_terme": long_term,
        "traffic_lights": {
            "court_terme": traffic_light(short_term.get("verdict")),
            "moyen_terme": traffic_light(mid_term.get("verdict")),
            "long_terme": traffic_light(long_term.get("verdict")),
        },
        "quick_summary": quick_summary,
    }


def build_profitability_signal(recommendation, global_score, ml_reliability, market_ctx, options_data, short_data):
    lights = recommendation.get("traffic_lights", {})
    light_values = [lights.get("court_terme"), lights.get("moyen_terme"), lights.get("long_terme")]
    green_count = sum(1 for x in light_values if x == "vert")
    red_count = sum(1 for x in light_values if x == "rouge")

    conf_vals = []
    for k in ["court_terme", "moyen_terme", "long_terme"]:
        c = recommendation.get(k, {}).get("confidence")
        if c is not None:
            conf_vals.append(float(c))
    avg_conf = round(sum(conf_vals) / len(conf_vals), 1) if conf_vals else 50

    # Penalise les phases de stress de marche pour eviter les faux positifs.
    risk_penalty = 0
    vix = market_ctx.get("vix")
    if vix is not None:
        if vix >= 32:
            risk_penalty += 12
        elif vix >= 25:
            risk_penalty += 6
    if short_data.get("squeeze_risk") in ["eleve", "extreme"]:
        risk_penalty += 5
    if options_data.get("signal") in ["fortement_baissier", "baissier"]:
        risk_penalty += 6

    base_score = (
        global_score * 0.55 +
        avg_conf * 0.20 +
        ml_reliability * 0.15 +
        (green_count * 100 / 3) * 0.10
    )
    score = round(max(0, min(100, base_score - risk_penalty)), 1)

    if green_count >= 2 and red_count == 0 and score >= 70:
        level = "fort_potentiel"
        label = "Fort potentiel de rentabilite"
        action = "PRIORITE"
    elif score >= 55 and red_count <= 1:
        level = "potentiel_modere"
        label = "Potentiel modere"
        action = "A SURVEILLER"
    else:
        level = "risque_ou_faible"
        label = "Potentiel limite / risque eleve"
        action = "PRUDENCE"

    return {
        "score": score,
        "level": level,
        "label": label,
        "action": action,
        "green_horizons": green_count,
        "red_horizons": red_count,
        "average_confidence": avg_conf,
        "risk_penalty": risk_penalty,
    }


def compute_solid_foundation_score(info, fund_score, market_ctx):
    checks = []
    pe = info.get("peRatio")
    if pe is None:
        checks.append(("Valorisation", 50, "P/E indisponible"))
    elif 0 < pe <= 28:
        checks.append(("Valorisation", 78, f"P/E raisonnable ({round(pe, 1)})"))
    elif pe <= 45:
        checks.append(("Valorisation", 58, f"P/E eleve ({round(pe, 1)})"))
    else:
        checks.append(("Valorisation", 35, f"P/E tres eleve ({round(pe, 1)})"))

    rg = info.get("revenueGrowth")
    if rg is None:
        checks.append(("Croissance CA", 50, "Croissance indisponible"))
    elif rg >= 0.12:
        checks.append(("Croissance CA", 82, f"Croissance forte ({round(rg*100,1)}%)"))
    elif rg >= 0.04:
        checks.append(("Croissance CA", 65, f"Croissance correcte ({round(rg*100,1)}%)"))
    elif rg >= 0:
        checks.append(("Croissance CA", 52, f"Croissance faible ({round(rg*100,1)}%)"))
    else:
        checks.append(("Croissance CA", 32, f"Croissance negative ({round(rg*100,1)}%)"))

    roe = info.get("roe")
    if roe is None:
        checks.append(("Rentabilite", 50, "ROE indisponible"))
    elif roe >= 0.18:
        checks.append(("Rentabilite", 85, f"ROE excellent ({round(roe*100,1)}%)"))
    elif roe >= 0.10:
        checks.append(("Rentabilite", 68, f"ROE solide ({round(roe*100,1)}%)"))
    elif roe >= 0:
        checks.append(("Rentabilite", 50, f"ROE moyen ({round(roe*100,1)}%)"))
    else:
        checks.append(("Rentabilite", 30, f"ROE negatif ({round(roe*100,1)}%)"))

    debt = info.get("debtToEquity")
    if debt is None:
        checks.append(("Dette", 50, "Dette/Cap indisponible"))
    elif debt <= 80:
        checks.append(("Dette", 80, f"Dette maitrisee ({round(debt,1)})"))
    elif debt <= 150:
        checks.append(("Dette", 58, f"Dette moderee ({round(debt,1)})"))
    else:
        checks.append(("Dette", 35, f"Dette elevee ({round(debt,1)})"))

    fcf = info.get("freeCashflow")
    checks.append(("Cash-flow", 75 if fcf and fcf > 0 else 30 if fcf and fcf < 0 else 50,
                   "FCF positif" if fcf and fcf > 0 else "FCF negatif" if fcf and fcf < 0 else "FCF indisponible"))

    macro = market_ctx.get("macro_score", 50)
    checks.append(("Regime Marche", 70 if macro >= 58 else 50 if macro >= 45 else 35,
                   f"Macro score {macro}/100"))

    raw = round(sum(c[1] for c in checks) / len(checks), 1)
    score = round(raw * 0.65 + fund_score * 0.35, 1)
    status = "bases_tres_solides" if score >= 72 else "bases_correctes" if score >= 58 else "bases_fragiles"
    return {
        "score": score,
        "status": status,
        "checks": [{"name": c[0], "score": c[1], "note": c[2]} for c in checks],
    }


def build_objective_analysis(info, recommendation, profitability, solid_foundation, market_ctx):
    strengths = []
    weaknesses = []
    risks = []

    if solid_foundation.get("score", 50) >= 70:
        strengths.append("Fondamentaux globalement solides")
    if profitability.get("score", 50) >= 70:
        strengths.append("Potentiel de rentabilite eleve")
    if recommendation.get("traffic_lights", {}).get("moyen_terme") == "vert":
        strengths.append("Momentum favorable a moyen terme")

    pe = info.get("peRatio")
    if pe and pe > 35:
        weaknesses.append("Valorisation exigeante (P/E eleve)")
    if (info.get("revenueGrowth") or 0) < 0:
        weaknesses.append("Croissance du chiffre d'affaires en recul")
    if solid_foundation.get("status") == "bases_fragiles":
        weaknesses.append("Base fondamentale insuffisamment robuste")

    if (market_ctx.get("vix") or 20) >= 30:
        risks.append("Volatilite de marche elevee (VIX)")
    if recommendation.get("traffic_lights", {}).get("court_terme") == "rouge":
        risks.append("Timing court terme defavorable")
    if profitability.get("risk_penalty", 0) >= 10:
        risks.append("Contexte risque penalise fortement le potentiel")

    verdict = "favorable" if profitability.get("score", 50) >= 70 and solid_foundation.get("score", 50) >= 65 else (
        "equilibre" if profitability.get("score", 50) >= 55 and solid_foundation.get("score", 50) >= 55 else "prudent"
    )
    return {
        "verdict_objectif": verdict,
        "strengths": strengths[:4],
        "weaknesses": weaknesses[:4],
        "risks": risks[:4],
        "note": "Analyse quantitative, non conseil financier.",
    }


TOP_UNIVERSE_DEFAULT = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX",
    "AMD", "INTC", "AVGO", "QCOM", "CRM", "ORCL", "IBM", "UBER",
    "JPM", "GS", "MS", "BAC", "V", "MA", "WMT", "JNJ",
    "XOM", "CVX", "MSTR", "COIN", "MARA", "RIOT",
]


def quick_profitability_scan(symbol, market_ctx):
    rows, rows_1h, rows_5m, rows_30m, info = fetch_data(symbol)
    prices = [r["close"] for r in rows]
    if len(prices) < 40:
        raise ValueError("Donnees insuffisantes")

    tech_sigs = compute_technical_signals(build_features(rows))
    fund_score = compute_fundamental_score(info)
    options_data = fetch_options_flow(symbol)
    short_data = fetch_short_interest(symbol, info)

    mc_h = {
        "1h": 1 / (252 * 6.5),
        "6h": 6 / (252 * 6.5),
        "24h": 1 / 252,
        "7d": 7 / 252,
        "1m": 21 / 252,
        "6m": 126 / 252,
    }
    mc_r = monte_carlo(prices, mc_h, n=2500)
    preds_adj = {"ensemble": mc_r}

    global_score = round((
        tech_sigs.get("score", 50) * 0.40 +
        fund_score * 0.25 +
        market_ctx.get("macro_score", 50) * 0.15 +
        options_data.get("score", 50) * 0.10 +
        short_data.get("score", 50) * 0.10
    ), 1)

    recommendation = build_horizon_recommendation(
        preds_adj,
        global_score,
        tech_sigs.get("score", 50),
        fund_score,
        50,  # sentiment neutre dans le scan rapide
    )
    profitability = build_profitability_signal(
        recommendation,
        global_score,
        50,  # fiabilite ML neutre dans le scan rapide
        market_ctx,
        options_data,
        short_data,
    )
    solid_foundation = compute_solid_foundation_score(info, fund_score, market_ctx)
    objective = build_objective_analysis(info, recommendation, profitability, solid_foundation, market_ctx)

    return {
        "symbol": symbol,
        "shortName": info.get("shortName", symbol),
        "price": info.get("currentPrice"),
        "dayChange": info.get("dayChange"),
        "global_score": global_score,
        "profitability": profitability,
        "solid_foundation": solid_foundation,
        "objective_analysis": objective,
        "recommendation": recommendation,
    }


def is_institutional_quality_candidate(scan_item):
    sf = scan_item.get("solid_foundation", {}).get("score", 50)
    pf = scan_item.get("profitability", {}).get("score", 50)
    rp = scan_item.get("profitability", {}).get("risk_penalty", 0)
    greens = scan_item.get("profitability", {}).get("green_horizons", 0)
    rec = scan_item.get("recommendation", {})
    lights = rec.get("traffic_lights", {})
    quick = rec.get("quick_summary", "")

    # Filtre strict: robustesse fondamentale + potentiel + risque contenu.
    if sf < 55:
        return False, "Solidite fondamentale insuffisante"
    if pf < 58:
        return False, "Potentiel de rentabilite trop faible"
    if rp >= 12:
        return False, "Risque de marche trop eleve"
    if greens < 1:
        return False, "Aucun horizon favorable"
    if lights.get("moyen_terme") == "rouge" and lights.get("long_terme") == "rouge":
        return False, "Moyen et long terme defavorables"
    if "EVITER" in quick and greens <= 1:
        return False, "Signal global trop defensif"
    return True, ""


# ══ ROUTE PRINCIPALE ══════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    body=request.get_json(force=True)
    query=(body.get("ticker") or "").strip()
    if not query: return jsonify({"error":"ticker requis"}), 400
    auto_clean_cache()
    ticker=resolve_ticker(query)
    print(f"[RESOLVE] '{query}' -> '{ticker}'")

    try:
        rows,rows_1h,rows_5m,rows_30m,info=fetch_data(ticker)
        prices=[r["close"] for r in rows]
        if len(prices)<15:
            return jsonify({"error":f"Donnees insuffisantes pour {ticker}"}), 400
        company_name=info.get("shortName","")

        # Fetch tout en parallèle — 11 sources simultanées
        with ThreadPoolExecutor(max_workers=11) as ex:
            f_macro    = ex.submit(fetch_market_context)
            f_news     = ex.submit(fetch_news, ticker)
            f_fh       = ex.submit(fetch_finnhub_all, ticker)
            f_vol      = ex.submit(fetch_realtime_volume, ticker, info)
            f_reddit   = ex.submit(fetch_reddit_sentiment, ticker, company_name)
            f_trends   = ex.submit(fetch_google_trends, ticker, company_name)
            f_backtest = ex.submit(run_backtest, ticker, rows, prices)
            f_options  = ex.submit(fetch_options_flow, ticker)
            f_insider  = ex.submit(fetch_insider_trading, ticker)
            f_short    = ex.submit(fetch_short_interest, ticker, info)
            f_sector   = ex.submit(fetch_sector_momentum, ticker, info)

            market_ctx    = f_macro.result(timeout=25)
            news          = f_news.result(timeout=10)
            fh_data       = f_fh.result(timeout=10)
            volume_data   = f_vol.result(timeout=15)
            reddit_data   = f_reddit.result(timeout=15)
            trends_data   = f_trends.result(timeout=15)
            backtest      = f_backtest.result(timeout=30)
            options_data  = f_options.result(timeout=20)
            insider_data  = f_insider.result(timeout=20)
            short_data    = f_short.result(timeout=10)
            sector_momentum = f_sector.result(timeout=20)

        # Earnings quality séparé (utilise Finnhub déjà fetchée)
        earnings_quality = fetch_earnings_quality(ticker)

        sentiment  = analyze_sentiment(ticker, news, market_ctx)
        df         = build_features(rows)
        tech_sigs  = compute_technical_signals(df)
        fund_score = compute_fundamental_score(info)

        mc_h={"1h":1/(252*6.5),"6h":6/(252*6.5),"24h":1/252,
              "7d":7/252,"1m":21/252,"6m":126/252}

        if ticker not in MODEL_CACHE:
            MODEL_CACHE[ticker]={"trained_at":time.time()}
        elif not cache_valid(ticker):
            MODEL_CACHE[ticker]["trained_at"]=time.time()

        mc_r  = monte_carlo(prices, mc_h)
        xgb_r = pred_xgb_multitf(ticker, rows, rows_1h, rows_5m, rows_30m)
        lstm_r= pred_lstm(ticker, prices)
        ens_r = make_ensemble(mc_r, xgb_r, lstm_r,
                              tech_sigs.get("score",50),
                              volume_data.get("volume_score",50),
                              options_data.get("score",50))

        preds_raw={"monte_carlo":mc_r,"xgboost":xgb_r,"lstm":lstm_r,"ensemble":ens_r}
        preds_raw=diversify_predictions(preds_raw, prices, info)
        preds_adj=apply_sentiment(preds_raw, sentiment.get("score",50), market_ctx)

        ai_txt=groq_masterclass_gs(
            ticker, info, preds_adj, sentiment, tech_sigs,
            market_ctx, fh_data, rows, volume_data,
            reddit_data, trends_data, backtest,
            options_data, insider_data, short_data,
            sector_momentum, earnings_quality)

        consensus={"strongBuy":0,"buy":0,"hold":0,"sell":0,"strongSell":0}
        if fh_data["recs"]:
            r0=fh_data["recs"][0]
            consensus={k:r0.get(k,0) for k in consensus}

        analyst_score   = compute_analyst_score(info, consensus)
        tot             = sum(consensus.values())
        consensus_score = round(
            (consensus.get("strongBuy",0)*2+consensus.get("buy",0))/(tot*2)*100,1
        ) if tot>0 else 50
        ml_reliability  = backtest.get("backtest_score",50)
        social_score    = round(
            reddit_data.get("score",50)*0.6+trends_data.get("interest_score",50)*0.4,1)

        metrics={
            "peRatio":info.get("peRatio"),"forwardPE":info.get("forwardPE"),
            "eps":info.get("eps"),"beta":info.get("beta"),
            "52WeekHigh":info.get("52WeekHigh"),"52WeekLow":info.get("52WeekLow"),
            "marketCap":info.get("marketCap"),"divYield":info.get("divYield"),
            "roe":info.get("roe"),"netMargin":info.get("netMargin"),
            "revenueGrowth":info.get("revenueGrowth"),"debtToEquity":info.get("debtToEquity"),
            "targetMean":info.get("targetMeanPrice"),"shortPct":info.get("shortPct"),
        }

        # Score global Goldman Sachs — 11 composantes
        global_score=round((
            analyst_score                          * 0.20 +
            tech_sigs.get("score",50)              * 0.15 +
            fund_score                             * 0.12 +
            sentiment.get("score",50)              * 0.10 +
            market_ctx.get("macro_score",50)       * 0.10 +
            options_data.get("score",50)           * 0.10 +
            insider_data.get("score",50)           * 0.08 +
            volume_data.get("volume_score",50)     * 0.06 +
            sector_momentum.get("relative_strength",50) * 0.05 +
            social_score                           * 0.02 +
            ml_reliability                         * 0.02
        ), 1)

        recommendation = build_horizon_recommendation(
            preds_adj,
            global_score,
            tech_sigs.get("score", 50),
            fund_score,
            sentiment.get("score", 50),
        )
        profitability = build_profitability_signal(
            recommendation,
            global_score,
            ml_reliability,
            market_ctx,
            options_data,
            short_data,
        )
        solid_foundation = compute_solid_foundation_score(info, fund_score, market_ctx)
        objective_analysis = build_objective_analysis(
            info, recommendation, profitability, solid_foundation, market_ctx
        )

        return jsonify({
            "symbol":          ticker,
            "query_original":  query,
            "shortName":       info.get("shortName",ticker),
            "timestamp":       datetime.utcnow().isoformat(),
            "quote":{"c":info["currentPrice"],"pc":info["previousClose"],
                     "dp":info.get("dayChange") or round(
                         (info["currentPrice"]-info["previousClose"])
                         /info["previousClose"]*100,2)},
            "metrics":         metrics,
            "info":            info,
            "predictions":     preds_adj,
            "consensus":       consensus,
            "consensus_score": consensus_score,
            "analyst_score":   analyst_score,
            "sentiment":       sentiment,
            "news":            news[:6],
            "technical":       tech_sigs,
            "market_context":  market_ctx,
            "volume":          volume_data,
            "reddit":          reddit_data,
            "trends":          trends_data,
            "backtest":        backtest,
            "options":         options_data,
            "insider":         insider_data,
            "short_interest":  short_data,
            "sector_momentum": sector_momentum,
            "earnings_quality":earnings_quality,
            "fundamental_score": fund_score,
            "ml_reliability":  ml_reliability,
            "social_score":    social_score,
            "global_score":    global_score,
            "recommendation":  recommendation,
            "profitability":   profitability,
            "solid_foundation": solid_foundation,
            "objective_analysis": objective_analysis,
            "earnings":        fh_data.get("earnings",[]),
            "peers":           fh_data.get("peers",[]),
            "ai_analysis":     ai_txt,
            "history":         rows[-90:],
            "intraday":        rows_5m[-200:] if rows_5m else [],
        })

    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":"ok","ml":ML_AVAILABLE,"lstm":LSTM_AVAILABLE,
        "groq":bool(GROQ_KEY),"finnhub":bool(FINNHUB_KEY),
        "trends":TRENDS_AVAILABLE,"cached":list(MODEL_CACHE.keys()),
        "version":"5.0-goldman-sachs"
    })


@app.route("/api/top10", methods=["POST"])
def top10():
    body = request.get_json(force=True) if request.data else {}
    raw = body.get("tickers", []) if isinstance(body, dict) else []
    strict_mode = bool(body.get("strict_mode", True)) if isinstance(body, dict) else True
    tickers = []
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, str) and t.strip():
                tickers.append(resolve_ticker(t.strip()))
    if not tickers:
        tickers = TOP_UNIVERSE_DEFAULT[:]
    tickers = list(dict.fromkeys(tickers))[:60]

    market_ctx = fetch_market_context()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(quick_profitability_scan, t, market_ctx): t for t in tickers}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append({"symbol": sym, "error": str(e)})

    filtered_out = []
    if strict_mode:
        kept = []
        for item in results:
            ok, reason = is_institutional_quality_candidate(item)
            if ok:
                kept.append(item)
            else:
                filtered_out.append({"symbol": item.get("symbol"), "reason": reason})
        results = kept

    ranked = sorted(results, key=lambda x: (
        x.get("profitability", {}).get("score", 0) * 0.65 +
        x.get("solid_foundation", {}).get("score", 0) * 0.35
    ), reverse=True)
    top_10 = ranked[:10]

    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "universe_size": len(tickers),
        "analyzed": len(results),
        "strict_mode": strict_mode,
        "top10": top_10,
        "filtered_out": filtered_out[:30],
        "errors": errors[:20],
    })


@app.route("/cache")
def cache_status():
    info={}
    for t,c in MODEL_CACHE.items():
        age=round((time.time()-c.get("trained_at",0))/60,1)
        info[t]={
            "age_minutes":age,
            "trained_at":datetime.fromtimestamp(c.get("trained_at",0)).strftime("%H:%M:%S"),
            "has_xgb":len(c.get("xgb_models",{}))>0,
            "has_lstm":c.get("lstm_model") is not None,
            "horizons_xgb":list(c.get("xgb_models",{}).keys()),
            "expires_in":round((CACHE_DURATION-(time.time()-c.get("trained_at",0)))/60,1)
        }
    return jsonify({"total":len(MODEL_CACHE),"tickers":info,
                    "cache_duration_h":CACHE_DURATION//3600})


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    body=request.get_json(force=True) if request.data else {}
    ticker=(body.get("ticker","") if body else "").upper()
    if ticker and ticker in MODEL_CACHE:
        del MODEL_CACHE[ticker]
        return jsonify({"message":f"Cache {ticker} supprime",
                        "remaining":list(MODEL_CACHE.keys())})
    elif not ticker:
        MODEL_CACHE.clear()
        return jsonify({"message":"Cache global vide"})
    return jsonify({"message":f"{ticker} non trouve",
                    "available":list(MODEL_CACHE.keys())})


@app.route("/")
def index():
    return jsonify({
        "message":"StockAI Pro v5.0 Goldman Sachs Edition",
        "lstm":LSTM_AVAILABLE,"ml":ML_AVAILABLE,
        "trends":TRENDS_AVAILABLE,
        "sources":["yfinance","finnhub","groq","reddit","google_trends",
                   "sec_edgar","options_flow","sector_momentum"],
        "cached":list(MODEL_CACHE.keys())
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
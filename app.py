import scipy.cluster.hierarchy as sch

if not hasattr(sch, "_LINKAGE_METHODS"):
    sch._LINKAGE_METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import datetime
import statsmodels.api as sm
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion, market_implied_prior_returns

import os
import time
import requests
import urllib.parse
import re
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = Flask(__name__)

DATA_CACHE = {}
MAX_CACHE_ITEMS = 10


def get_market_data(tickers, benchmark_ticker='^NSEI'):
    unique_tickers = sorted(list(set(tickers)))
    cache_key = f"{benchmark_ticker}:{','.join(unique_tickers)}"
    if cache_key in DATA_CACHE:
        return DATA_CACHE[cache_key]

    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=10 * 365)

    series_dict = {}
    mcaps = {}

    for ticker in unique_tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
            if not hist.empty and 'Close' in hist:
                s = hist['Close'].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                if not s.empty:
                    series_dict[ticker] = s
                    mcaps[ticker] = 1e10
        except Exception as e:
            print(f"Price fetch failed for {ticker}: {e}")

    price_data = pd.DataFrame(series_dict).dropna(how='all').ffill().dropna()

    bench_hist = yf.Ticker(benchmark_ticker).history(start=start_date, end=end_date, auto_adjust=True)
    if bench_hist.empty or 'Close' not in bench_hist:
        raise ValueError(f"Could not fetch benchmark data for {benchmark_ticker}")

    bench_data = bench_hist['Close'].dropna()
    bench_data.index = pd.to_datetime(bench_data.index).tz_localize(None)

    returns = price_data.pct_change().dropna(how='all').dropna()
    bench_returns = bench_data.pct_change().dropna()
    returns, bench_returns = returns.align(bench_returns, join='inner', axis=0)

    if len(DATA_CACHE) >= MAX_CACHE_ITEMS:
        oldest_key = next(iter(DATA_CACHE))
        del DATA_CACHE[oldest_key]

    result = (price_data, returns, bench_returns, mcaps, bench_data)
    DATA_CACHE[cache_key] = result
    return result


def calc_advanced_metrics(returns, weights_dict, bench_returns, risk_free_rate=0.07):
    active_weights = {k: float(v) for k, v in weights_dict.items() if k in returns.columns and float(v) > 0}
    if not active_weights:
        return {
            "1Y Return (%)": "N/A", "3Y Return (%)": "N/A", "5Y Return (%)": "N/A", "10Y Return (%)": "N/A",
            "Risk/Vol (%)": 0, "Sharpe Ratio": 0, "Info Ratio": 0, "Treynor Ratio": 0, "Jensen's Alpha (%)": 0
        }

    total_w = sum(active_weights.values())
    tickers = list(active_weights.keys())
    weights_array = np.array([active_weights[t] / total_w for t in tickers])

    port_returns = returns[tickers].dot(weights_array)

    ann_ret = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    bench_ret = bench_returns.mean() * 252

    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol != 0 else 0
    tracking_error = (port_returns - bench_returns).std() * np.sqrt(252)
    info_ratio = (ann_ret - bench_ret) / tracking_error if tracking_error != 0 else 0

    X = sm.add_constant(bench_returns)
    beta = sm.OLS(port_returns, X).fit().params.iloc[1]

    treynor = (ann_ret - risk_free_rate) / beta if beta != 0 else 0
    alpha = ann_ret - (risk_free_rate + beta * (bench_ret - risk_free_rate))

    def cagr(years):
        days = int(years * 252)
        if len(port_returns) < days:
            return "N/A"
        cum = (1 + port_returns.iloc[-days:]).prod()
        return round(((cum ** (1 / years)) - 1) * 100, 2)

    return {
        "1Y Return (%)": cagr(1),
        "3Y Return (%)": cagr(3),
        "5Y Return (%)": cagr(5),
        "10Y Return (%)": cagr(10),
        "Risk/Vol (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Info Ratio": round(info_ratio, 2),
        "Treynor Ratio": round(treynor, 4),
        "Jensen's Alpha (%)": round(alpha * 100, 2)
    }


@app.route('/')
def index():
    return render_template('index.html')


def get_custom_mvo_weights(action, metric, returns, bench_returns, rf):
    num_assets = returns.shape[1]

    days = len(returns)
    if '1 Year' in metric:
        days = 252
    elif '3 Year' in metric:
        days = 252 * 3
    elif '5 Year' in metric:
        days = 252 * 5
    elif '10 Year' in metric:
        days = 252 * 10

    opt_returns = returns.tail(days)
    opt_bench = bench_returns.tail(days) if bench_returns is not None else None

    mean_rets = opt_returns.mean() * 252
    cov_matrix = opt_returns.cov() * 252
    bench_ret = opt_bench.mean() * 252 if opt_bench is not None else 0

    betas = np.zeros(num_assets)
    if opt_bench is not None and not opt_bench.empty:
        comb = pd.concat([opt_returns, opt_bench], axis=1).cov()
        bench_var = comb.iloc[-1, -1]
        if bench_var > 0:
            for i, col in enumerate(opt_returns.columns):
                betas[i] = comb.loc[col, opt_bench.name] / bench_var

    def obj_fun(weights):
        port_ret = np.sum(weights * mean_rets)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if metric == 'Risk':
            val = port_vol
        elif metric == 'Sharpe Ratio':
            val = (port_ret - rf) / port_vol if port_vol > 0 else 0
        elif 'Return' in metric:
            val = port_ret
        elif metric == "Jensen's Alpha":
            port_beta = np.sum(weights * betas)
            val = port_ret - (rf + port_beta * (bench_ret - rf))
        elif metric == 'Treynor Ratio':
            port_beta = np.sum(weights * betas)
            val = (port_ret - rf) / port_beta if port_beta != 0 else 0
        elif metric == 'Information Ratio':
            if opt_bench is not None:
                port_daily_rets = np.dot(opt_returns.values, weights)
                tracking_diff = port_daily_rets - opt_bench.values
                te = np.std(tracking_diff, ddof=1) * np.sqrt(252)
                val = (port_ret - bench_ret) / te if te > 0 else 0
            else:
                val = 0
        else:
            val = port_ret

        return -val if action == 'Maximize' else val

    init_guess = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

    result = sco.minimize(obj_fun, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return {opt_returns.columns[i]: result.x[i] for i in range(num_assets)}
    else:
        return {opt_returns.columns[i]: init_guess[i] for i in range(num_assets)}


@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    universe_tickers = [t.strip() for t in data.get('tickers', '').split(',') if t.strip()]
    custom_views = data.get('views', {})
    custom_mfs = data.get('custom_mfs', {})
    rf = float(data.get('risk_free_rate', 7.0)) / 100.0
    action = data.get('action', 'Maximize')
    metric = data.get('metric', 'Sharpe Ratio')

    if not universe_tickers:
        return jsonify({"error": "No tickers provided"}), 400

    all_tickers = set(universe_tickers)
    for mf_name, mf_weights in custom_mfs.items():
        all_tickers.update(mf_weights.keys())
    all_tickers = list(all_tickers)

    try:
        price_data, returns, bench_returns, mcaps, bench_data = get_market_data(all_tickers)
    except Exception as e:
        return jsonify({"error": f"Market data fetch failed: {str(e)}"}), 502

    valid_universe_tickers = [t for t in universe_tickers if t in price_data.columns]

    if not valid_universe_tickers:
        return jsonify({"error": "Failed to retrieve historical data for the requested tickers."}), 400

    univ_price_data = price_data[valid_universe_tickers]
    univ_returns = returns[valid_universe_tickers]
    univ_mcaps = {t: mcaps.get(t, 1e10) for t in valid_universe_tickers}

    valid_custom_views = {k: v for k, v in custom_views.items() if k in valid_universe_tickers}

    mu = expected_returns.mean_historical_return(univ_price_data)
    S = risk_models.sample_cov(univ_price_data)

    results = {"weights": {}, "metrics": {}}

    # 1. Process Multiple Mutual Funds
    for mf_name, mf_weights in custom_mfs.items():
        results['weights'][mf_name] = mf_weights
        results['metrics'][mf_name] = calc_advanced_metrics(returns, mf_weights, bench_returns, rf)

    # 2. Advanced MVO Custom Scipy Engine
    raw_mvo_w = get_custom_mvo_weights(action, metric, univ_returns, bench_returns, rf)
    mvo_w = {k: round(v * 100, 2) for k, v in raw_mvo_w.items()}

    results['weights']['MVO'] = mvo_w
    results['metrics']['MVO'] = calc_advanced_metrics(univ_returns, mvo_w, bench_returns, rf)

    # 3. Hierarchical Risk Parity (HRP)
    try:
        hrp = HRPOpt(univ_returns)
        clean_hrp = hrp.optimize(linkage_method="single")
        hrp_w = {k: round(v * 100, 2) for k, v in clean_hrp.items()}
        results['weights']['HRP'] = hrp_w
        results['metrics']['HRP'] = calc_advanced_metrics(univ_returns, hrp_w, bench_returns, rf)
    except Exception as e:
        print("HRP Error:", e)

    # 4. Black-Litterman Model
    try:
        delta = market_implied_risk_aversion(bench_data) if bench_data is not None and not bench_data.empty else 2.5
        prior = market_implied_prior_returns(pd.Series(univ_mcaps), delta, S, rf)

        if valid_custom_views:
            bl = BlackLittermanModel(S, pi=prior, absolute_views=valid_custom_views)
        else:
            bl = BlackLittermanModel(S, pi=prior)

        ef_bl = EfficientFrontier(bl.bl_returns(), bl.bl_cov())

        if action == 'Minimize' and metric == 'Risk':
            ef_bl.min_volatility()
        else:
            ef_bl.max_sharpe(risk_free_rate=rf)

        clean_bl = ef_bl.clean_weights()
        bl_w = {k: round(v * 100, 2) for k, v in clean_bl.items()}
        results['weights']['Black-Litterman'] = bl_w
        results['metrics']['Black-Litterman'] = calc_advanced_metrics(univ_returns, bl_w, bench_returns, rf)
    except Exception as e:
        print("Black-Litterman Error:", e)

    return jsonify(results)


@app.route('/recalculate', methods=['POST'])
def recalculate():
    data = request.json
    manual_weights = data.get('weights', {})
    rf = float(data.get('risk_free_rate', 7.0)) / 100.0
    tickers = list(manual_weights.keys())

    try:
        _, returns, bench_returns, _, _ = get_market_data(tickers)
    except Exception as e:
        return jsonify({"error": f"Market data fetch failed: {str(e)}"}), 502

    new_metrics = calc_advanced_metrics(returns, manual_weights, bench_returns, rf)
    return jsonify({"metrics": new_metrics})


@app.route('/search_fund', methods=['POST'])
def search_fund():
    data = request.json
    raw_input = data.get('fund_name', '').strip()

    if not raw_input:
        return jsonify({"error": "Fund name or URL slug required."}), 400

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }

        if raw_input.startswith("http://") or raw_input.startswith("https://"):
            target_url = raw_input
        else:
            slug = raw_input.lower().strip().replace("&", "and").replace(" ", "-")
            if slug.startswith("mutual-funds/"):
                target_url = f"https://groww.in/{slug}"
            else:
                target_url = f"https://groww.in/mutual-funds/{slug}"

        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        html_resp = session.get(target_url, headers=headers, timeout=10)

        if html_resp.status_code != 200:
            return jsonify({"error": f"Could not load Groww page (HTTP {html_resp.status_code})."}), 404

        soup = BeautifulSoup(html_resp.text, 'html.parser')

        h1_tag = soup.find('h1')
        actual_fund_name = h1_tag.text.strip() if h1_tag else raw_input.replace("-", " ").title()

        holdings_section = soup.find('section', {'id': 'holdingsContainer'})
        if not holdings_section:
            holdings_section = soup.find('div', class_=re.compile(r'holdings_tableContainer', re.I))

        if not holdings_section:
            return jsonify({"error": f"Could not locate the Holdings Section on the page for {actual_fund_name}."}), 404

        holdings_table = holdings_section.find('table', class_=re.compile(r'tb10Table', re.I))
        if not holdings_table:
            holdings_table = holdings_section.find('table')

        if not holdings_table:
            return jsonify({"error": f"Could not locate the holding table inside the Holdings Section for {actual_fund_name}."}), 404

        portfolio_weights = {}
        unmapped = []

        top_stocks = {
            # Nifty 50 & Major Banks/Financials
            "HDFC BANK": "HDFCBANK.NS", "ICICI BANK": "ICICIBANK.NS", "STATE BANK OF INDIA": "SBIN.NS",
            "AXIS BANK": "AXISBANK.NS", "KOTAK MAHINDRA": "KOTAKBANK.NS", "INDUSIND BANK": "INDUSINDBK.NS",
            "BAJAJ FINANCE": "BAJFINANCE.NS", "BAJAJ FINSERV": "BAJAJFINSV.NS", "CHOLAMANDALAM": "CHOLAFIN.NS",
            "MUTHOOT FINANCE": "MUTHOOTFIN.NS", "SBI LIFE": "SBILIFE.NS", "HDFC LIFE": "HDFCLIFE.NS",
            "ICICI PRUDENTIAL": "ICICIPRULI.NS", "ICICI LOMBARD": "ICICIGI.NS", "HDFC ASSET": "HDFCAMC.NS",

            # Defence & Aerospace
            "HINDUSTAN AERONAUTICS": "HAL.NS", "BHARAT ELECTRONICS": "BEL.NS", "BHARAT DYNAMICS": "BDL.NS",
            "MAZAGON DOCK": "MAZDOCK.NS", "COCHIN SHIPYARD": "COCHINSHIP.NS", "GARDEN REACH": "GRSE.NS",
            "DATA PATTERNS": "DATAPATTNS.NS", "ASTRA MICROWAVE": "ASTRAMICRO.NS", "ZEN TECHNOLOGIES": "ZENTEC.NS",

            # PSUs, Railways & Power
            "POWER FINANCE": "PFC.NS", "REC LTD": "RECLTD.NS", "IRFC": "IRFC.NS", "RVNL": "RVNL.NS",
            "IRCON": "IRCON.NS", "RITES": "RITES.NS", "PUNJAB NATIONAL BANK": "PNB.NS",
            "BANK OF BARODA": "BANKBARODA.NS", "CANARA BANK": "CANBK.NS", "UNION BANK": "UNIONBANK.NS",
            "INDIAN BANK": "INDIANB.NS", "BANK OF INDIA": "BANKINDIA.NS", "LIFE INSURANCE CORPORATION": "LICI.NS",
            "BHEL": "BHEL.NS", "SAIL": "SAIL.NS", "NMDC": "NMDC.NS", "HINDUSTAN COPPER": "HINDCOPPER.NS",
            "NATIONAL ALUMINIUM": "NATIONALUM.NS", "SJVN": "SJVN.NS", "NHPC": "NHPC.NS",
            "HUDCO": "HUDCO.NS", "IREDA": "IREDA.NS", "OIL AND NATURAL GAS": "ONGC.NS",

            # IT, Tech & Telecom
            "TATA CONSULTANCY": "TCS.NS", "INFOSYS": "INFY.NS", "HCL TECHNOLOGIES": "HCLTECH.NS",
            "WIPRO": "WIPRO.NS", "TECH MAHINDRA": "TECHM.NS", "LTIMINDTREE": "LTIM.NS",
            "PERSISTENT SYSTEMS": "PERSISTENT.NS", "COFORGE": "COFORGE.NS", "MPHASIS": "MPHASIS.NS",
            "BHARTI AIRTEL": "BHARTIARTL.NS", "INDUS TOWERS": "INDUSTOWER.NS",

            # Energy, Oil & Gas
            "RELIANCE": "RELIANCE.NS", "NTPC": "NTPC.NS", "POWER GRID": "POWERGRID.NS",
            "COAL INDIA": "COALINDIA.NS", "TATA POWER": "TATAPOWER.NS", "ADANI GREEN": "ADANIGREEN.NS",
            "ADANI POWER": "ADANIPOWER.NS", "BPCL": "BPCL.NS", "GAIL": "GAIL.NS", "INDIAN OIL": "IOC.NS",

            # FMCG & Retail
            "HINDUSTAN UNILEVER": "HINDUNILVR.NS", "ITC LTD": "ITC.NS", "NESTLE": "NESTLEIND.NS",
            "BRITANNIA": "BRITANNIA.NS", "TATA CONSUMER": "TATACONSUM.NS", "GODREJ CONSUMER": "GODREJCP.NS",
            "AVENUE SUPERMARTS": "DMART.NS", "TRENT": "TRENT.NS", "TITAN": "TITAN.NS",
            "DABUR": "DABUR.NS", "MARICO": "MARICO.NS", "COLGATE": "COLPAL.NS", "ZOMATO": "ZOMATO.NS",

            # Auto & Manufacturing
            "TATA MOTORS": "TATAMOTORS.NS", "MAHINDRA & MAHINDRA": "M&M.NS", "MARUTI SUZUKI": "MARUTI.NS",
            "BAJAJ AUTO": "BAJAJ-AUTO.NS", "HERO MOTOCORP": "HEROMOTOCO.NS", "EICHER MOTORS": "EICHERMOT.NS",
            "TVS MOTOR": "TVSMOTOR.NS", "ASHOK LEYLAND": "ASHOKLEY.NS", "BOSCH": "BOSCHLTD.NS",

            # Infrastructure, Cement & Metals
            "LARSEN & TOUBRO": "LT.NS", "ULTRATECH CEMENT": "ULTRACEMCO.NS", "GRASIM": "GRASIM.NS",
            "AMBUJA CEMENTS": "AMBUJACEM.NS", "SHREE CEMENT": "SHREECEM.NS", "ACC LTD": "ACC.NS",
            "TATA STEEL": "TATASTEEL.NS", "JSW STEEL": "JSWSTEEL.NS", "HINDALCO": "HINDALCO.NS",
            "VEDANTA": "VEDL.NS", "JINDAL STEEL": "JINDALSTEL.NS",

            # Pharma & Healthcare
            "SUN PHARMACEUTICAL": "SUNPHARMA.NS", "DR. REDDY": "DRREDDY.NS", "CIPLA": "CIPLA.NS",
            "DIVI'S LABORATORIES": "DIVISLAB.NS", "APOLLO HOSPITALS": "APOLLOHOSP.NS",
            "LUPIN": "LUPIN.NS", "ZYDUS": "ZYDUSLIFE.NS", "AUROBINDO": "AUROPHARMA.NS",

            # Others
            "ASIAN PAINTS": "ASIANPAINT.NS", "PIDILITE": "PIDILITIND.NS", "IRCTC": "IRCTC.NS",
            "HAVELLS": "HAVELLS.NS", "VOLTAS": "VOLTAS.NS", "INTERGLOBE AVIATION": "INDIGO.NS",
            "JIO FINANCIAL": "JIOFIN.NS"
        }

        tbody = holdings_table.find('tbody')
        rows = tbody.find_all('tr') if tbody else holdings_table.find_all('tr')

        for row in rows:
            if row.find('th'):
                continue

            comp_td = row.find('td', class_=re.compile(r'holdings_companyName', re.I))
            weight_td = row.find('td', class_=re.compile(r'right-align', re.I))

            cols = row.find_all('td')
            if not comp_td and len(cols) >= 1:
                comp_td = cols[0]
            if not weight_td and len(cols) >= 2:
                weight_td = cols[-1]

            if comp_td and weight_td:
                company_name = comp_td.text.strip()
                weight_str = weight_td.text.strip().replace('%', '').replace(',', '')

                if not company_name or "View All" in company_name:
                    continue

                try:
                    weight = float(weight_str)
                except ValueError:
                    continue

                if weight == 0 or "TREPS" in company_name.upper() or "MARGIN" in company_name.upper():
                    continue

                clean_name = company_name.upper()
                mapped_symbol = None
                for key, val in top_stocks.items():
                    if key in clean_name:
                        mapped_symbol = val
                        break

                if mapped_symbol:
                    portfolio_weights[mapped_symbol] = weight
                    continue

                time.sleep(0.6)
                yf_search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(company_name + ' NSE')}&quotesCount=1"

                try:
                    resp = session.get(yf_search_url, headers=headers, timeout=5)
                    yf_resp = resp.json() if resp.status_code == 200 else {}
                    quotes = yf_resp.get('quotes', [])
                    if quotes and 'symbol' in quotes[0]:
                        symbol = quotes[0]['symbol']
                        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                            symbol += '.NS'
                        portfolio_weights[symbol] = weight
                    else:
                        unmapped.append(company_name)
                except Exception as e:
                    print(f"Failed to map {company_name}: {e}")
                    unmapped.append(company_name)

        session.close()

        if not portfolio_weights:
            if unmapped:
                return jsonify({"error": f"Extracted {len(unmapped)} companies (like '{unmapped[0]}') from the table, but Yahoo Finance blocked all requests. You are being rate-limited. Wait 1 minute and try again."}), 500
            else:
                return jsonify({"error": f"Found table for '{actual_fund_name}', but the HTML <tbody> was completely empty. Groww is blocking the rows from loading."}), 500

        return jsonify({
            "fund_name": actual_fund_name,
            "weights": portfolio_weights,
            "unmapped": unmapped
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
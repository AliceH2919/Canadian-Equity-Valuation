from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

TICKERS = {
    "CNQ.TO": {"name": "Canadian Natural Resources", "sector": "Energy"},
    "ENB.TO": {"name": "Enbridge", "sector": "Energy / Infrastructure"},
    "CNR.TO": {"name": "Canadian National Railway", "sector": "Industrials"},
    "CP.TO": {"name": "Canadian Pacific Kansas City", "sector": "Industrials"},
    "RY.TO": {"name": "Royal Bank of Canada", "sector": "Financials"},
    "TD.TO": {"name": "Toronto-Dominion Bank", "sector": "Financials"},
}

RISK_FREE_RATE = 0.035
EQUITY_RISK_PREMIUM = 0.055
DEFAULT_BETA = 1.00
TAX_RATE = 0.26
TERMINAL_GROWTH = 0.025
DCF_GROWTH = 0.05
PROJECTION_YEARS = 5
DDM_GROWTH = 0.04
INITIAL_CAPITAL = 10_000
RISK_FREE_BACKTEST = 0.035


def _value(obj, keys, default=np.nan):
    for key in keys:
        try:
            value = obj.get(key)
            if value is not None and pd.notna(value):
                return float(value)
        except Exception:
            pass
    return default


def _last_value(frame, labels):
    if frame is None or frame.empty:
        return np.nan
    for label in labels:
        if label in frame.index:
            series = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])
    return np.nan


def get_company_data(ticker):
    t = yf.Ticker(ticker)

    try:
        info = t.info
    except Exception:
        info = {}

    try:
        income = t.income_stmt
    except Exception:
        income = pd.DataFrame()

    try:
        cashflow = t.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    try:
        balance = t.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    try:
        price = float(t.fast_info.get("last_price", np.nan))
    except Exception:
        price = _value(info, ["currentPrice", "regularMarketPrice"])

    market_cap = _value(info, ["marketCap"])
    try:
        fast_cap = float(t.fast_info.get("market_cap", np.nan))
        if not np.isfinite(market_cap):
            market_cap = fast_cap
    except Exception:
        pass

    shares = _value(info, ["sharesOutstanding", "impliedSharesOutstanding"])
    if not np.isfinite(shares) and market_cap > 0 and price > 0:
        shares = market_cap / price

    revenue = _last_value(income, ["Total Revenue", "Operating Revenue"])
    ebit = _last_value(income, ["EBIT", "Operating Income"])
    net_income = _last_value(income, ["Net Income", "Net Income Common Stockholders"])
    ebitda = _last_value(income, ["EBITDA", "Normalized EBITDA"])

    ocf = _last_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _last_value(cashflow, ["Capital Expenditure", "Capital Expenditure Reported"])
    if np.isfinite(capex):
        capex = abs(capex)

    cash = _last_value(balance, [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents"
    ])
    debt = _last_value(balance, [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation"
    ])

    fcf = ocf - capex if np.isfinite(ocf) else np.nan

    try:
        divs = t.dividends
        dividend = float(divs.tail(252).sum()) if not divs.empty else np.nan
    except Exception:
        dividend = np.nan

    eps = _value(info, ["trailingEps"])
    book_value = _value(info, ["bookValue"])

    pe = price / eps if price > 0 and eps > 0 else np.nan
    pb = price / book_value if price > 0 and book_value > 0 else np.nan
    ps = market_cap / revenue if market_cap > 0 and revenue > 0 else np.nan

    ev = market_cap + (debt if np.isfinite(debt) else 0) - (cash if np.isfinite(cash) else 0)
    ev_ebitda = ev / ebitda if ev > 0 and ebitda > 0 else np.nan

    beta = _value(info, ["beta", "beta3Year"], DEFAULT_BETA)
    if not np.isfinite(beta):
        beta = DEFAULT_BETA

    return {
        "ticker": ticker,
        "name": TICKERS[ticker]["name"],
        "sector": TICKERS[ticker]["sector"],
        "price": price,
        "market_cap": market_cap,
        "shares": shares,
        "revenue": revenue,
        "ebit": ebit,
        "net_income": net_income,
        "ebitda": ebitda,
        "ocf": ocf,
        "capex": capex,
        "fcf": fcf,
        "cash": cash,
        "debt": debt,
        "eps": eps,
        "book_value": book_value,
        "dividend": dividend,
        "beta": beta,
        "pe": pe,
        "pb": pb,
        "ps": ps,
        "ev_ebitda": ev_ebitda,
    }


def dcf_details(d, growth=DCF_GROWTH, terminal_growth=TERMINAL_GROWTH, wacc_override=None):
    if not all(np.isfinite(d[k]) for k in ["shares", "revenue", "ebit"]) or d["shares"] <= 0:
        return np.nan

    margin = np.clip(d["ebit"] / d["revenue"], 0.05, 0.35)
    fcff0 = d["revenue"] * margin * (1 - TAX_RATE) * 0.70

    if np.isfinite(d["fcf"]) and d["fcf"] > 0:
        fcff0 = 0.70 * d["fcf"] + 0.30 * fcff0

    ke = RISK_FREE_RATE + d["beta"] * EQUITY_RISK_PREMIUM
    kd = 0.045
    debt = d["debt"] if np.isfinite(d["debt"]) and d["debt"] > 0 else 0
    equity = d["market_cap"] if np.isfinite(d["market_cap"]) and d["market_cap"] > 0 else 0
    capital = equity + debt

    wacc = (
        (equity / capital) * ke +
        (debt / capital) * kd * (1 - TAX_RATE)
        if capital > 0 else ke
    )

    if wacc_override is not None:
        wacc = wacc_override

    wacc = max(wacc, terminal_growth + 0.02)

    pv = 0
    fcff = fcff0
    for year in range(1, PROJECTION_YEARS + 1):
        fcff *= 1 + growth
        pv += fcff / ((1 + wacc) ** year)

    terminal = fcff * (1 + terminal_growth) / (wacc - terminal_growth)
    enterprise_value = pv + terminal / ((1 + wacc) ** PROJECTION_YEARS)

    net_debt = (
        (d["debt"] if np.isfinite(d["debt"]) else 0)
        - (d["cash"] if np.isfinite(d["cash"]) else 0)
    )

    return (enterprise_value - net_debt) / d["shares"]


def dcf_value(d):
    return dcf_details(d)


def sector_medians(df):
    medians = {}
    for sector in df["sector"].unique():
        peer = df[df["sector"] == sector]
        medians[sector] = {
            col: peer[col].replace([np.inf, -np.inf], np.nan).median()
            for col in ["pe", "pb", "ps", "ev_ebitda"]
        }
    return medians


def cca_value(d, df):
    med = sector_medians(df)[d["sector"]]
    implied = []

    if np.isfinite(med["pe"]) and d["eps"] > 0:
        implied.append(med["pe"] * d["eps"])

    if np.isfinite(med["pb"]) and d["book_value"] > 0:
        implied.append(med["pb"] * d["book_value"])

    if np.isfinite(med["ps"]) and d["shares"] > 0:
        implied.append(med["ps"] * d["revenue"] / d["shares"])

    return float(np.mean(implied)) if implied else np.nan


def ddm_value(d):
    if not np.isfinite(d["dividend"]) or d["dividend"] <= 0:
        return np.nan

    ke = RISK_FREE_RATE + d["beta"] * EQUITY_RISK_PREMIUM
    if ke <= DDM_GROWTH:
        return np.nan

    return d["dividend"] * (1 + DDM_GROWTH) / (ke - DDM_GROWTH)


def build_valuation_table(raw):
    df = pd.DataFrame(raw)
    df["dcf_fair_value"] = [dcf_value(x.to_dict()) for _, x in df.iterrows()]
    df["cca_fair_value"] = [cca_value(x.to_dict(), df) for _, x in df.iterrows()]
    df["ddm_fair_value"] = [ddm_value(x.to_dict()) for _, x in df.iterrows()]

    methods = ["dcf_fair_value", "cca_fair_value", "ddm_fair_value"]
    df["fair_value"] = df[methods].mean(axis=1, skipna=True)
    df["upside_pct"] = (df["fair_value"] / df["price"] - 1) * 100

    def signal(x):
        if not np.isfinite(x):
            return "Insufficient Data"
        if x >= 15:
            return "Potentially Undervalued"
        if x <= -15:
            return "Potentially Overvalued"
        return "Approximately Fair Value"

    df["signal"] = df["upside_pct"].apply(signal)
    return df.sort_values("upside_pct", ascending=False)


def make_allocation(df):
    picks = df[df["signal"] == "Potentially Undervalued"].copy()
    if len(picks) < 2:
        picks = df.head(2).copy()

    picks = picks.head(4).copy()
    weights = np.repeat(1 / len(picks), len(picks))
    weights = np.minimum(weights, 0.50)
    weights /= weights.sum()

    picks["weight"] = weights
    picks["investment"] = INITIAL_CAPITAL * picks["weight"]

    return picks[[
        "ticker", "name", "fair_value", "price",
        "upside_pct", "signal", "weight", "investment"
    ]]


def download_prices(tickers):
    data = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = tickers

    return close.ffill().dropna()


def backtest(selected_tickers):
    tickers = selected_tickers + ["^GSPTSE"]
    close = download_prices(tickers)

    if close.empty:
        return pd.DataFrame()

    returns = close.pct_change().dropna()

    # Equal-weighted daily rebalanced basket.
    portfolio_return = returns[selected_tickers].mean(axis=1)
    benchmark_return = returns["^GSPTSE"]

    result = pd.DataFrame({
        "Value Portfolio": (1 + portfolio_return).cumprod(),
        "S&P/TSX Composite": (1 + benchmark_return).cumprod(),
    })

    return result


def performance_metrics(result):
    if result.empty:
        return {}

    daily = result.pct_change().dropna()

    metrics = {}
    for col in result.columns:
        cumulative = result[col].iloc[-1] - 1
        vol = daily[col].std() * np.sqrt(252)
        excess = daily[col].mean() * 252 - RISK_FREE_BACKTEST
        sharpe = excess / vol if vol > 0 else np.nan

        running_max = result[col].cummax()
        drawdown = result[col] / running_max - 1

        metrics[col] = {
            "Cumulative Return": cumulative,
            "Annualized Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": drawdown.min(),
            "Best Day": daily[col].max(),
            "Worst Day": daily[col].min(),
        }

    return metrics


def plot_valuation(df):
    p = df.dropna(subset=["price", "fair_value"]).copy()
    if p.empty:
        return

    x = np.arange(len(p))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width/2, p["price"], width, label="Market Price")
    ax.bar(x + width/2, p["fair_value"], width, label="Composite Fair Value")
    ax.set_xticks(x)
    ax.set_xticklabels(p["ticker"], rotation=30)
    ax.set_ylabel("CAD / share")
    ax.set_title("Market Price vs. Composite Fair Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "valuation_comparison.png", dpi=180)
    plt.close(fig)


def plot_peer_multiples(df):
    cols = ["pe", "pb", "ev_ebitda"]
    p = df.set_index("ticker")[cols].replace([np.inf, -np.inf], np.nan)

    fig, ax = plt.subplots(figsize=(11, 6))
    p.plot(kind="bar", ax=ax)
    ax.set_title("Selected Canadian Companies: Valuation Multiples")
    ax.set_ylabel("Multiple")
    ax.set_xlabel("Ticker")
    ax.legend(title="Multiple")
    fig.tight_layout()
    fig.savefig(OUTPUT / "peer_multiples.png", dpi=180)
    plt.close(fig)


def plot_dcf_sensitivity(d):
    waccs = np.arange(0.07, 0.111, 0.01)
    growths = np.arange(0.01, 0.041, 0.005)

    matrix = []
    for g in growths:
        row = []
        for w in waccs:
            row.append(dcf_details(d, terminal_growth=g, wacc_override=w))
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(waccs)))
    ax.set_xticklabels([f"{x:.0%}" for x in waccs])
    ax.set_yticks(range(len(growths)))
    ax.set_yticklabels([f"{x:.1%}" for x in growths])
    ax.set_xlabel("WACC")
    ax.set_ylabel("Terminal Growth")
    ax.set_title(f"DCF Sensitivity — {d['ticker']}")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"${value:.0f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Fair Value / Share (CAD)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "dcf_sensitivity.png", dpi=180)
    plt.close(fig)

    pd.DataFrame(
        matrix,
        index=[f"g={x:.1%}" for x in growths],
        columns=[f"WACC={x:.0%}" for x in waccs]
    ).to_csv(OUTPUT / "dcf_sensitivity.csv")


def plot_backtest(result):
    if result.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    result.plot(ax=ax)
    ax.set_title("1-Year Value Portfolio vs. S&P/TSX Composite")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "portfolio_performance.png", dpi=180)
    plt.close(fig)


def plot_drawdown(result):
    if result.empty:
        return

    dd = result / result.cummax() - 1

    fig, ax = plt.subplots(figsize=(11, 6))
    dd.plot(ax=ax)
    ax.set_title("Portfolio and Benchmark Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT / "drawdown.png", dpi=180)
    plt.close(fig)


def save_research_report(df, allocation, result):
    lines = [
        "# Investment Research Summary",
        "",
        "## Valuation ranking",
        "",
        df[[
            "ticker", "name", "price", "dcf_fair_value",
            "cca_fair_value", "ddm_fair_value",
            "fair_value", "upside_pct", "signal"
        ]].to_markdown(index=False),
        "",
        "## Hypothetical $10,000 portfolio",
        "",
        allocation.to_markdown(index=False),
        "",
    ]

    metrics = performance_metrics(result)
    if metrics:
        metric_rows = []
        for name, m in metrics.items():
            metric_rows.append({
                "Asset": name,
                "Cumulative Return": f"{m['Cumulative Return']:.1%}",
                "Annualized Volatility": f"{m['Annualized Volatility']:.1%}",
                "Sharpe Ratio": f"{m['Sharpe Ratio']:.2f}",
                "Max Drawdown": f"{m['Max Drawdown']:.1%}",
                "Best Day": f"{m['Best Day']:.1%}",
                "Worst Day": f"{m['Worst Day']:.1%}",
            })

        lines += [
            "## 1-year backtest",
            "",
            pd.DataFrame(metric_rows).to_markdown(index=False),
            "",
        ]

    valid = df.dropna(subset=["upside_pct"])
    if not valid.empty:
        best = valid.iloc[0]
        worst = valid.iloc[-1]
        lines += [
            "## Conclusion",
            "",
            f"- **Most potentially undervalued:** {best['name']} ({best['ticker']}) with estimated upside of {best['upside_pct']:.1f}%.",
            f"- **Most potentially overvalued:** {worst['name']} ({worst['ticker']}) with estimated upside of {worst['upside_pct']:.1f}%.",
            "- The $10,000 portfolio is allocated across the highest-ranked value candidates.",
            "",
            "> This conclusion is model-driven and sensitive to assumptions; it is not investment advice.",
        ]

    (OUTPUT / "research_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    print("Downloading Canadian equity data...\n")

    raw = []
    for ticker in TICKERS:
        try:
            print(f"Loading {ticker}...")
            raw.append(get_company_data(ticker))
        except Exception as exc:
            print(f"Warning: {ticker} failed: {exc}")

    if not raw:
        raise RuntimeError("No data downloaded.")

    df = build_valuation_table(raw)

    valuation_cols = [
        "ticker", "name", "sector", "price",
        "pe", "pb", "ps", "ev_ebitda",
        "dcf_fair_value", "cca_fair_value", "ddm_fair_value",
        "fair_value", "upside_pct", "signal"
    ]
    df[valuation_cols].to_csv(OUTPUT / "valuation_summary.csv", index=False)

    allocation = make_allocation(df)
    allocation.to_csv(OUTPUT / "portfolio_allocation.csv", index=False)

    selected = allocation["ticker"].tolist()
    result = backtest(selected)

    if not result.empty:
        result.to_csv(OUTPUT / "portfolio_backtest.csv")

    plot_valuation(df)
    plot_peer_multiples(df)

    best_dcf = df.dropna(subset=["dcf_fair_value"])
    if not best_dcf.empty:
        plot_dcf_sensitivity(best_dcf.iloc[0].to_dict())

    plot_backtest(result)
    plot_drawdown(result)
    save_research_report(df, allocation, result)

    print("\n" + "=" * 70)
    print("INVESTMENT CONCLUSION")
    print("=" * 70)

    valid = df.dropna(subset=["upside_pct"])
    if not valid.empty:
        best = valid.iloc[0]
        worst = valid.iloc[-1]
        print(f"Potentially undervalued: {best['name']} ({best['ticker']}) — {best['upside_pct']:.1f}%")
        print(f"Potentially overvalued: {worst['name']} ({worst['ticker']}) — {worst['upside_pct']:.1f}%")

    print("\n$10,000 allocation:")
    for _, row in allocation.iterrows():
        print(f"  {row['ticker']}: {row['weight']:.0%} = ${row['investment']:,.0f}")

    metrics = performance_metrics(result)
    if metrics:
        print("\n1-year performance:")
        for name, m in metrics.items():
            print(
                f"  {name}: return {m['Cumulative Return']:.1%}, "
                f"volatility {m['Annualized Volatility']:.1%}, "
                f"max drawdown {m['Max Drawdown']:.1%}"
            )

    print("\nOutputs saved to outputs/")

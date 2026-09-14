# Canadian Equity Valuation & Value Portfolio

> **Python | yfinance | pandas | NumPy | Matplotlib | Financial Analysis**

A portfolio-ready equity research project that evaluates Canadian public companies using **DCF**, **Comparable Company Analysis (CCA)**, and **Dividend Discount Model (DDM)**, then builds a hypothetical **$10,000 value portfolio** and compares its historical 1-year performance with the **S&P/TSX Composite**.

## Why this project

This project turns a basic stock-screening exercise into a complete investment research workflow:

**Market data → Financial metrics → Valuation → Investment signal → Portfolio construction → Backtest**

It demonstrates skills relevant to equity research, investment analysis, risk analysis, and financial data analytics.

## Research universe

- Canadian Natural Resources (`CNQ.TO`)
- Enbridge (`ENB.TO`)
- Canadian National Railway (`CNR.TO`)
- Canadian Pacific Kansas City (`CP.TO`)
- Royal Bank of Canada (`RY.TO`)
- Toronto-Dominion Bank (`TD.TO`)

## Valuation methods

### DCF — Discounted Cash Flow

The model estimates intrinsic value using projected FCFF:

`FCFF = EBIT × (1 − Tax Rate) + D&A − Capex − ΔNWC`

A five-year forecast is followed by a Gordon Growth terminal value.

The model also creates a **WACC × terminal-growth sensitivity table**, showing how fair value changes when assumptions change.

### Comparable Company Analysis

The model calculates:

- P/E
- P/B
- P/S
- EV/EBITDA

It uses peer/sector median multiples to estimate implied value.

### Dividend Discount Model

For companies with meaningful dividends:

`P₀ = D₁ / (Ke − g)`

where `Ke` is cost of equity and `g` is long-run dividend growth.

### Composite fair value

Available valuation methods are combined into a transparent composite fair value estimate.

**Potentially Undervalued:** upside ≥ 15%  
**Approximately Fair Value:** between -15% and +15%  
**Potentially Overvalued:** upside ≤ -15%

## Portfolio construction

A hypothetical **$10,000** portfolio is constructed from the strongest value candidates, subject to a 50% maximum position size.

The output includes:

- Weight
- Dollar allocation
- Fair value
- Current price
- Estimated upside

## Backtest

The selected basket is evaluated over the trailing 1-year period and compared with:

**S&P/TSX Composite (`^GSPTSE`)**

The project reports:

- Cumulative return
- Relative performance
- Maximum drawdown
- Annualized volatility
- Sharpe ratio (simple risk-free assumption)
- Best/worst day

### Important backtest limitation

This is a **historical performance analysis of the current model-selected basket**, not a point-in-time historical strategy backtest. Current valuation inputs should not be interpreted as information that was necessarily available one year ago.

## Outputs

Running the project creates:

```text
outputs/
├── valuation_summary.csv
├── portfolio_allocation.csv
├── portfolio_backtest.csv
├── peer_multiples.png
├── dcf_sensitivity.png
├── portfolio_performance.png
├── drawdown.png
└── research_summary.md
```

## Run

```bash
pip install -r requirements.txt
python run_model.py
```

Optional notebook:

```bash
jupyter notebook
```

## Example questions answered

- Which company appears most undervalued?
- Which company appears most overvalued?
- What is the implied fair value per share?
- How would I allocate $10,000?
- Did the value basket outperform the TSX?
- How much downside risk did the portfolio experience?

## Technical stack

| Tool | Use |
|---|---|
| Python | Research pipeline |
| yfinance | Market / company data |
| pandas | Data cleaning and financial metrics |
| NumPy | Numerical calculations |
| Matplotlib | Research visualizations |
| Jupyter | Reproducible analysis |

## Disclaimer

This is an educational equity research project and is **not investment advice**. Valuation results depend heavily on assumptions and on the availability/quality of Yahoo Finance data. Real investment research would require audited filings, management guidance, company-specific forecasts, and more detailed capital-structure analysis.

## Resume version

**Canadian Equity Valuation & Value Portfolio — Python**

- Built a multi-method equity valuation model for six Canadian companies using DCF, comparable-company analysis, and dividend discount models.
- Automated market-data retrieval and valuation analysis with `yfinance`, `pandas`, and NumPy; calculated fair value, valuation upside, peer multiples, and DCF sensitivity scenarios.
- Constructed a hypothetical $10,000 value portfolio and evaluated 1-year performance, volatility, drawdown, and relative return versus the S&P/TSX Composite using Matplotlib.

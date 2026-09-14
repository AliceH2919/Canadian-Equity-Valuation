# Canadian Equity Valuation — Walkthrough

## 1. Objective

Build a transparent valuation framework for Canadian equities and test a hypothetical value portfolio against the S&P/TSX Composite.

## 2. Workflow

1. Pull company and market data with `yfinance`.
2. Clean financial statement data with `pandas`.
3. Calculate P/E, P/B, P/S and EV/EBITDA.
4. Estimate DCF fair value.
5. Estimate comparable-company fair value.
6. Estimate DDM fair value where dividends are available.
7. Combine available methods.
8. Rank companies by estimated upside.
9. Allocate a hypothetical $10,000.
10. Backtest the basket against `^GSPTSE`.
11. Analyze volatility and drawdown.

## 3. Interpretation

A stock is not automatically a good investment because one model says it is cheap. The strongest research conclusion is one where:

- multiple valuation methods point in a similar direction;
- operating fundamentals support the valuation;
- assumptions are realistic;
- downside risk is understood.

## 4. Limitations

The DCF is simplified and uses explicit assumptions. Financial institutions also require sector-specific valuation approaches because debt and operating liabilities behave differently from industrial companies.

The backtest is not point-in-time. It evaluates the current selected basket over the trailing year, so it should not be presented as evidence that the strategy would have been known or selected one year earlier.

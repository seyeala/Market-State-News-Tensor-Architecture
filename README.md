# Market-State News-Tensor Architecture

This repository implements a point-in-time research toolkit for the normalized market-state and news-tensor architecture described in the design report. The project focuses on leakage-safe feature generation, return/trading-label construction, and cost-aware validation before adding higher-capacity models or live trading integrations.

## Initial MVP

The first implementation sprint covers:

1. Python package scaffold.
2. Point-in-time market, quote, news, event, feature, and label schemas.
3. Point-in-time validation utilities.
4. Intraday market feature generation.
5. Basic cost model.
6. Fixed-horizon and triple-barrier labels.

## Development

```bash
python -m pytest
```

## Safety

This repository is a technical research implementation. It is not financial advice and does not assert that any model, signal, or strategy is profitable.

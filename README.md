# Market-State News-Tensor Architecture

This repository implements a point-in-time research toolkit for the normalized market-state and news-tensor architecture described in the design report. The project focuses on leakage-safe feature generation, return/trading-label construction, and cost-aware validation before adding higher-capacity models or live trading integrations.

## Quickstart

Install the project in a Python environment with the development dependencies, then run the test suite from the repository root:

```bash
python -m pytest
```

## Current Implementation Status

| Area | Status | Notes |
| --- | --- | --- |
| Scaffold | Complete | Python package scaffold, `src/` layout, pytest configuration, and module folders are in place. |
| Schemas | Partial | Market bars, quotes, raw news, ex-ante events, ex-post outcomes, feature rows, labels, and experiment records are defined. Universe and tensor-specific schemas are not yet implemented. |
| Point-in-time utilities | Complete | Availability checks, valid-row filtering, and ex-post field guards are implemented. |
| Market features | Complete for MVP | Log price selection, intraday returns, realized volatility, range, volume, VWAP deviation, spread, time-of-day features, and rolling market tensors are implemented. |
| Costs | Complete for MVP | A simple return-unit transaction-cost model covers spread, fees, slippage, expected impact, and volatility/size impact. |
| Labels | Complete for MVP | Fixed-horizon future returns, trading classes, and triple-barrier labels are implemented with label windows. |
| Context features | Not started | Cross-sectional, universe, macro, sector, and other non-event context feature builders remain to be added. |
| Event tensors | Not started | Event ontology, event loading/extraction, staleness, state tensor updates, and reliability gates remain to be added. |
| Models | Not started | Modeling modules are placeholders; no baseline, neural, or ensemble models are implemented yet. |
| Validation | Partial | Point-in-time checks and cost helpers exist, but walk-forward splitting, purging/embargo, metrics, and experiment validation are still pending. |
| CLI | Not started | No command-line entry points are defined yet. |
| Examples | Not started | Example datasets, notebooks, and runnable workflow scripts are still pending. |

## MVP Workflow

1. **Build market features** from point-in-time `MarketBar` inputs with `build_market_feature_rows`, then assemble rolling model-ready windows with `build_market_tensor`.
2. **Create labels** from aligned log-price points using fixed-horizon return/class labels or triple-barrier labels, keeping `label_start` and `label_end` for later purged validation.
3. **Validate outputs** by checking that every feature row was available by its decision time, rejecting ex-post event fields from live payloads, and applying cost-aware label thresholds before any model training.

## Safety

This repository is a technical research implementation. It is not financial advice and does not assert that any model, signal, or strategy is profitable.

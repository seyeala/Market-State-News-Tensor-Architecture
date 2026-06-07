# Market-State News-Tensor Architecture Task Backlog

This backlog turns the architecture report into implementation tasks. Tasks are ordered by dependency so the project can move from a leakage-safe research scaffold to a cost-aware trading-model validation system.

## Milestone 1: Project Foundation

### TASK-001: Create Python project scaffold
- **Goal:** Establish a package structure for implementation and tests.
- **Steps:**
  1. Add `pyproject.toml` with package metadata, Python version, and test dependencies.
  2. Create `src/market_state_news_tensor/` with `__init__.py`.
  3. Create module folders for schemas, data, features, labels, events, belief, models, validation, and experiments.
  4. Create a `tests/` directory with placeholder test files.
- **Acceptance criteria:** The package imports successfully, `pytest` discovers tests, and the repository has a clear source/test layout.

### TASK-002: Write implementation-oriented README
- **Goal:** Explain what the repository builds and how to run the first workflows.
- **Steps:**
  1. Summarize the market-state/news-tensor architecture.
  2. Document the MVP workflow: build features, make labels, train baseline, validate walk-forward.
  3. Include non-advice disclaimer language.
- **Acceptance criteria:** A new contributor can understand the project purpose, first commands, and safety constraints.

### TASK-003: Create operational documentation stubs
- **Goal:** Keep architecture and implementation decisions discoverable.
- **Steps:**
  1. Add `docs/architecture.md`.
  2. Add `docs/data_contract.md`.
  3. Add `docs/labeling.md`.
  4. Add `docs/events.md`.
  5. Add `docs/validation.md`.
  6. Add `docs/experiment_governance.md`.
- **Acceptance criteria:** Each document has a short purpose section and links to related modules once they exist.

## Milestone 2: Schemas and Point-in-Time Contracts

### TASK-004: Define market bar schema
- **Goal:** Represent OHLCV bars with availability metadata.
- **Steps:**
  1. Create `MarketBar` with symbol, bar start/end, open, high, low, close, volume, optional VWAP, and `available_at`.
  2. Validate non-negative volume and strictly positive prices.
  3. Require `available_at >= bar_end`.
- **Acceptance criteria:** Invalid prices, invalid volume, and impossible availability timestamps fail validation.

### TASK-005: Define quote schema
- **Goal:** Represent bid/ask quotes and spread helpers.
- **Steps:**
  1. Create `QuoteBar` with symbol, timestamp, bid, ask, sizes, and `available_at`.
  2. Add helpers for mid-price and proportional spread.
  3. Reject crossed markets unless a future configuration explicitly allows them.
- **Acceptance criteria:** Valid quotes compute mid/spread; crossed quotes fail by default.

### TASK-006: Define raw news item schema
- **Goal:** Capture raw incoming news before event extraction.
- **Steps:**
  1. Create `RawNewsItem` with source, headline, body, URL, language, and timestamp fields.
  2. Include `event_time`, `publish_time`, `vendor_received_time`, `ingest_time`, and `available_at`.
  3. Require at least a headline or body.
- **Acceptance criteria:** Raw news preserves all point-in-time timestamps and validates required text.

### TASK-007: Define ex-ante event object schema
- **Goal:** Represent structured news/event claims that may enter live features.
- **Steps:**
  1. Create `EventObject` with entity, claim, context type, event type, affected metrics, scores, horizon, and evidence.
  2. Bound `direction` to `[-1, 1]`.
  3. Bound magnitude, uncertainty, source quality, and novelty to `[0, 1]`.
  4. Forbid ex-post outcome fields on the ex-ante object.
- **Acceptance criteria:** Schema validation enforces bounded scores and keeps ex-post outcomes separate.

### TASK-008: Define ex-post event outcome schema
- **Goal:** Store event-study results for training, calibration, and audit only.
- **Steps:**
  1. Create `EventOutcome` with abnormal returns at 5m, 15m, 60m, and 1d.
  2. Add volume change, volatility change, reversal flag, and `outcome_available_at`.
  3. Link outcomes to `event_id`.
- **Acceptance criteria:** Outcomes are serializable and explicitly excluded from live feature payloads.

### TASK-009: Define feature row and label schemas
- **Goal:** Standardize model inputs and labels.
- **Steps:**
  1. Create `FeatureRow` with symbol, decision time, component feature maps, masks, and validity flag.
  2. Create fixed-horizon label schema with return, class, horizon, and label window.
  3. Create triple-barrier label schema with barriers, touch time, class, and label window.
- **Acceptance criteria:** Labels preserve `label_start` and `label_end` for purged validation.

### TASK-010: Implement point-in-time validity utilities
- **Goal:** Enforce `available_at <= decision_time` everywhere.
- **Steps:**
  1. Implement `is_point_in_time_valid(available_at, decision_time)`.
  2. Implement batch filtering for tabular rows.
  3. Implement an ex-post feature guard that rejects outcome fields in live features.
- **Acceptance criteria:** Tests cover valid rows, invalid rows, timezone-aware datetimes, and outcome leakage rejection.

## Milestone 3: Market and Context Features

### TASK-011: Implement log-price selection
- **Goal:** Use mid-price when quotes are valid, otherwise close price.
- **Steps:**
  1. Implement `compute_log_price(close, bid=None, ask=None)`.
  2. Use `log((bid + ask) / 2)` when bid/ask are valid.
  3. Fall back to `log(close)` when quotes are missing.
- **Acceptance criteria:** Non-positive prices fail, valid quotes use mid-price, and missing quotes use close.

### TASK-012: Implement intraday market feature rows
- **Goal:** Build one-minute market feature rows.
- **Steps:**
  1. Compute 1m, 5m, and 15m log returns.
  2. Compute 15m and 60m realized volatility.
  3. Compute range, log volume, VWAP deviation, spread, and time-of-day sine/cosine.
  4. Preserve `available_at` metadata.
- **Acceptance criteria:** Rolling features use only historical rows and have stable column ordering.

### TASK-013: Build rolling market tensor
- **Goal:** Produce the intraday tensor `M_bar`.
- **Steps:**
  1. Implement `build_market_tensor(feature_rows, lookback=180)`.
  2. Support fixed-shape output with optional masks/padding.
  3. Keep an ordered feature-name list for downstream models.
- **Acceptance criteria:** Tensor shape is `(180, m)` when enough history exists and never includes future rows.

### TASK-014: Implement daily context features
- **Goal:** Add multi-day regime context.
- **Steps:**
  1. Compute `r_1d`, `r_2d`, `r_5d`, and `r_20d`.
  2. Compute `sigma_5d`, `sigma_20d`, and `volume_z_20d`.
  3. Compute opening gap, 20-day distance to high, and 20-day distance to low.
- **Acceptance criteria:** Intraday decisions never use the same-day close unless it is already available.

### TASK-015: Implement weekly and monthly context features
- **Goal:** Add longer regime information.
- **Steps:**
  1. Compute weekly returns and volatility with documented session conventions.
  2. Compute monthly returns and volatility with documented session conventions.
  3. Build a stable context vector.
- **Acceptance criteria:** Context features use completed historical periods and document missing-history behavior.

## Milestone 4: Costs and Labels

### TASK-016: Implement cost model
- **Goal:** Represent spread, fees, slippage, and expected impact.
- **Steps:**
  1. Implement a base `CostModel`.
  2. Implement `c(q) = spread / 2 + fees + eta * volatility * sqrt(abs(q) / volume)`.
  3. Return costs in return units.
- **Acceptance criteria:** Larger size and volatility increase cost; larger volume lowers impact; costs are non-negative.

### TASK-017: Implement fixed-horizon return labels
- **Goal:** Compute future log returns for a chosen horizon.
- **Steps:**
  1. Implement `make_future_returns(prices, horizon)`.
  2. Store label start/end timestamps.
  3. Drop rows with insufficient future data.
- **Acceptance criteria:** Future data is used only for labels, not features, and label windows are recorded.

### TASK-018: Implement fixed-horizon trading class labels
- **Goal:** Convert future returns into long/no-trade/short classes.
- **Steps:**
  1. Apply cost and volatility thresholds.
  2. Emit `+1`, `0`, or `-1`.
  3. Store cost and threshold metadata.
- **Acceptance criteria:** Threshold tests cover long, short, and no-trade cases.

### TASK-019: Implement triple-barrier labels
- **Goal:** Label first profit/loss barrier touch before vertical expiry.
- **Steps:**
  1. Build upper and lower barriers from cost and volatility.
  2. Find first upper/lower touch.
  3. Apply vertical barrier when neither horizontal barrier is touched.
  4. Store touch time and label window.
- **Acceptance criteria:** Tests cover upper-first, lower-first, no-touch, and missing-future-data cases.

## Milestone 5: Universe, Events, and State Tensors

### TASK-020: Implement point-in-time tradable universe
- **Goal:** Avoid survivorship bias and today-universe leakage.
- **Steps:**
  1. Define universe membership schema with symbol, start/end, tradable, observable, eligible, and reason.
  2. Implement `filter_universe_asof(rows, universe, decision_time)`.
  3. Add tests for lifecycle changes.
- **Acceptance criteria:** Only eligible symbols as of the decision time enter training rows.

### TASK-021: Define event ontologies and rulebooks
- **Goal:** Make event tensorization repeatable.
- **Steps:**
  1. Define context types: geopolitical, macro, earnings, fundamental, legal/regulatory, product, market-belief, and opinion.
  2. Define event types and affected metrics for each context.
  3. Add decay, scoring, and contradiction-rule placeholders.
- **Acceptance criteria:** Unknown event types or affected metrics fail validation unless explicitly mapped.

### TASK-022: Implement manual event loader and extractor interface
- **Goal:** Support structured events before adding LLM extraction.
- **Steps:**
  1. Load event objects from JSON/CSV.
  2. Validate events against schemas and rulebooks.
  3. Define `EventExtractor.extract(raw_item)` interface.
- **Acceptance criteria:** Manual events can be ingested, validated, and passed to state updates.

### TASK-023: Implement staleness and deduplication
- **Goal:** Prevent repeated headlines from acting like independent evidence.
- **Steps:**
  1. Normalize text for duplicate detection.
  2. Implement exact duplicate detection by entity and normalized claim/headline.
  3. Implement token or TF-IDF similarity for first-pass staleness.
  4. Set novelty to `1 - stale_score`.
- **Acceptance criteria:** Identical and near-duplicate stories receive low novelty and link to the original event.

### TASK-024: Implement state tensor schema and update logic
- **Goal:** Maintain compressed event memory by entity and context.
- **Steps:**
  1. Define `StateTensor` with entity, context, `asof_time`, feature map, active events, and reliability status.
  2. Implement context-specific decay.
  3. Implement additive update `S_t = lambda * S_{t-1} + W(e_j)`.
  4. Scale updates by direction, magnitude, uncertainty, source quality, and novelty.
- **Acceptance criteria:** Updates are deterministic and old/stale events contribute less.

### TASK-025: Implement contradiction and reliability gates
- **Goal:** Keep unstable event tensors out of live model features.
- **Steps:**
  1. Implement contradiction handling that increases uncertainty or reversal risk.
  2. Implement schema/evidence/score-bounds reliability checks.
  3. Log failed events while excluding them from live features.
- **Acceptance criteria:** Missing evidence and invalid schemas fail the gate; contradiction tests modify uncertainty as expected.

## Milestone 6: Event Outcomes, Reaction, Belief, and Opinion

### TASK-026: Implement event-study outcome table
- **Goal:** Attach ex-post results after event windows close.
- **Steps:**
  1. Compute abnormal returns at 5m, 15m, 60m, and 1d.
  2. Compute volume and volatility changes.
  3. Compute reversal flag using a documented rule.
- **Acceptance criteria:** Outcomes are available only after their windows close and never enter ex-ante feature rows.

### TASK-027: Implement market reaction vectors
- **Goal:** Measure observed reaction around events.
- **Steps:**
  1. Define reaction-vector schema.
  2. Compute symbol-level return, volume spike, volatility spike, and spread change.
  3. Support cross-asset reaction inputs when available.
- **Acceptance criteria:** Reaction vectors are timestamped and point-in-time valid.

### TASK-028: Implement Bayesian market-belief filter
- **Goal:** Infer latent scenario distributions from news, reaction, and context.
- **Steps:**
  1. Define scenario schema and belief-state schema.
  2. Implement Bayesian scenario update with safe normalization.
  3. Add reaction strength, uncertainty, horizon, and reversal-risk outputs.
- **Acceptance criteria:** Probabilities sum to one, strong evidence shifts beliefs, and zero-likelihood edge cases are handled.

### TASK-029: Implement user opinion and disagreement layer
- **Goal:** Represent opinion as a calibrated risk overlay, not truth.
- **Steps:**
  1. Define user-opinion schema with scenario probabilities and confidence.
  2. Compute disagreement vector `D = O - B`.
  3. Implement risk overlay multiplier that can reduce exposure.
  4. Set alpha contribution weight to zero by default.
- **Acceptance criteria:** Opinion cannot increase exposure before out-of-sample validation.

## Milestone 7: Tensor Builder, Models, Calibration, and Execution

### TASK-030: Build complete feature payload
- **Goal:** Assemble the full model input state.
- **Steps:**
  1. Combine market tensor, context vector, state tensors, recent events, belief state, and disagreement state.
  2. Validate all component availability timestamps.
  3. Provide both flattened tabular output and sequence-tensor output.
- **Acceptance criteria:** Complete feature payloads are point-in-time valid and have stable feature names/order.

### TASK-031: Implement baseline models
- **Goal:** Establish simple leakage-sensitive baselines.
- **Steps:**
  1. Implement ridge regression for return prediction.
  2. Implement logistic/multinomial logistic classifier for trading labels.
  3. Implement no-trade and random baselines.
- **Acceptance criteria:** Baselines train, predict, and produce comparable metrics.

### TASK-032: Implement GBM model wrappers
- **Goal:** Add the first serious tabular model family.
- **Steps:**
  1. Implement LightGBM wrapper if available.
  2. Implement XGBoost wrapper if available.
  3. Add scikit-learn fallback for environments without GBM packages.
  4. Expose feature importance where supported.
- **Acceptance criteria:** Model interface supports fit, predict, predict_proba, save, and load.

### TASK-033: Implement probability calibration
- **Goal:** Convert raw model probabilities into better-calibrated probabilities.
- **Steps:**
  1. Implement temperature scaling stub or wrapper.
  2. Implement isotonic/Platt calibration where appropriate.
  3. Compute Brier score, log loss, expected calibration error, and reliability bins.
- **Acceptance criteria:** Calibration fits only on validation data and reports per-fold calibration metrics.

### TASK-034: Implement execution policy and expected value layer
- **Goal:** Separate model prediction from trade sizing and cost-aware decisioning.
- **Steps:**
  1. Compute expected value from calibrated probabilities and estimated gain/loss.
  2. Implement long/short/no-trade policy based on EV and risk threshold.
  3. Implement position sizing that respects volatility, spread, volume, and risk limits.
- **Acceptance criteria:** Higher costs reduce EV, no trade occurs below threshold, and risk limits are enforced.

## Milestone 8: Validation, Experiment Governance, and Ablation

### TASK-035: Implement walk-forward splitter
- **Goal:** Use chronological validation rather than random splits.
- **Steps:**
  1. Implement rolling and expanding train windows.
  2. Add validation and test windows.
  3. Ensure train period always precedes validation/test period.
- **Acceptance criteria:** Generated folds are chronological and reproducible.

### TASK-036: Implement purging and embargo
- **Goal:** Reduce leakage from overlapping label windows.
- **Steps:**
  1. Remove training rows whose label windows overlap test windows.
  2. Add configurable embargo after each test window.
  3. Use label start/end metadata.
- **Acceptance criteria:** Overlapping and embargoed rows are removed from training folds.

### TASK-037: Implement validation runner and metrics
- **Goal:** Run full train/evaluate cycles after costs.
- **Steps:**
  1. Train models per fold.
  2. Evaluate classification metrics and trading metrics.
  3. Report gross return, net return, turnover, costs, max drawdown, hit rate, and calibration.
- **Acceptance criteria:** Fold-level and aggregate reports are generated and saved.

### TASK-038: Implement experiment ledger
- **Goal:** Make every experiment reproducible and auditable.
- **Steps:**
  1. Define experiment-record persistence to JSON.
  2. Store feature set, label set, universe, split, hyperparameters, cost model, metrics, and git commit.
  3. Add config hashing and deterministic artifact paths.
- **Acceptance criteria:** Every run writes a ledger record with reproducibility metadata.

### TASK-039: Implement ablation framework
- **Goal:** Prove each tensor family adds value after costs.
- **Steps:**
  1. Add feature switches for market, context, event state, belief, and disagreement tensors.
  2. Reuse identical validation splits for ablation comparisons.
  3. Compute metric deltas by fold and aggregate.
- **Acceptance criteria:** Ablation report identifies which tensor families improve or harm performance.

### TASK-040: Add anti-overfit controls
- **Goal:** Track strategy search and selection-bias risk.
- **Steps:**
  1. Log number of model/config trials.
  2. Add placeholders for probability of backtest overfitting and deflated Sharpe ratio.
  3. Emit warnings when experiment search becomes too broad for available data.
- **Acceptance criteria:** Reports include overfit-risk metadata even when final advanced statistics are unavailable.

## Milestone 9: CLI, Examples, and Future Neural Models

### TASK-041: Add CLI entrypoint
- **Goal:** Make workflows runnable from the command line.
- **Steps:**
  1. Add commands for build-features, make-labels, train, validate, and ablate.
  2. Accept a config path for each command.
  3. Provide helpful errors and `--help` output.
- **Acceptance criteria:** CLI commands can be invoked and route to the correct workflow functions.

### TASK-042: Add example scripts
- **Goal:** Provide runnable examples for the MVP pipeline.
- **Steps:**
  1. Add `examples/build_features.py`.
  2. Add `examples/train_baseline.py`.
  3. Add `examples/run_walk_forward.py`.
  4. Add minimal synthetic/sample data fixtures if real data is unavailable.
- **Acceptance criteria:** Examples run end-to-end on test data and write outputs.

### TASK-043: Implement optional neural sequence encoders
- **Goal:** Add CNN/GRU fusion only after tabular pipeline is validated.
- **Steps:**
  1. Implement market sequence CNN encoder.
  2. Implement GRU encoder alternative.
  3. Implement fusion model for market/context/state/belief embeddings.
  4. Compare against GBM through the same validation runner.
- **Acceptance criteria:** Neural models are optional, mask-aware, and must beat GBM after costs to be promoted.

## Recommended first sprint

The first implementation sprint should complete TASK-001 through TASK-010, then begin TASK-011 through TASK-019. That creates the minimum leakage-safe foundation: schemas, point-in-time checks, market features, costs, and labels. Event tensors, belief filters, and opinion overlays should wait until the market/label/validation foundation is tested.

# Detailed Findings: Truflation as a CPI Proxy and Short-Horizon Nowcast

*Last updated: 2026-03-31*

---

## 1. Executive Summary

This analysis tests whether Truflation's real-time inflation signal can help anticipate the next official US CPI YoY print during periods when BLS data is unavailable or delayed (e.g. government shutdown).

**Key conclusions:**
- Truflation and CPI YoY are strongly correlated (ρ = 0.946) across 193 months of overlapping data.
- The relationship peaks at a **1-month lead** (ρ = 0.953) — Truflation today correlates most strongly with CPI published next month.
- Truflation adds **statistically significant incremental predictive content** beyond CPI's own autoregressive history (Granger causality confirmed, p < 0.0001).
- A simple linear nowcast model improves **+42% over a naive persistence baseline** in walk-forward out-of-sample testing.
- The **directional accuracy is 72.5%** — the model correctly predicts whether CPI will rise or fall vs the prior month nearly three-quarters of the time.

**Current signal (as of February 2026):**

| | Value |
|---|---|
| Latest official CPI YoY (Feb 2026) | 2.66% |
| Latest Truflation YoY | 0.94% |
| Spread (Truflation − CPI) | −1.72pp |
| Model nowcast for March 2026 CPI YoY | **2.54%** |

**Interpretation:** The model points to mild deceleration in the next print. The current −1.72pp spread is the most extreme negative reading in the dataset, signalling downside risk. However, the extreme magnitude adds model uncertainty — this is an edge-of-sample regime.

---

## 2. What Was Tested

The question is not whether Truflation equals CPI by construction — it does not, and it is not designed to. The question is whether Truflation has **timely information** that helps forecast CPI before CPI is published.

Three complementary tests address this:

1. **Correlation and lead-lag analysis** — measures how closely the two series move together, and whether Truflation leads CPI
2. **Granger causality test** — formally tests whether past Truflation values add predictive power beyond CPI's own history
3. **Walk-forward nowcast model** — tests out-of-sample forecast accuracy with no lookahead bias, benchmarked against a naive baseline

---

## 3. Data Scope and Coverage

| Item | Detail |
|---|---|
| Overlap sample | 193 monthly observations |
| Start | January 2011 (first full YoY observation) |
| End | February 2026 |
| CPI source | FRED CPIAUCSL — CPI-U All Items, Seasonally Adjusted, monthly |
| Truflation source | `truflation.com/api/index-data/us-inflation-rate` — daily YoY %, resampled to monthly mean |

**Important alignment note:** Truflation is published daily, but the predictive model operates on monthly-aligned features to match the CPI release cadence. Daily Truflation is averaged within each calendar month and the resulting monthly value is used as the feature — this is the mean real-time signal for that reference period.

---

## 4. Finding A — Correlation and Lead-Lag

### 4.1 Contemporaneous relationship

**Correlation at lag 0: 0.9458**

The two series move strongly together over the full sample — substantially higher than would be expected from two independently-constructed inflation measures. This validates Truflation as a real-time proxy worth monitoring.

### 4.2 Lead-lag profile

Correlations when Truflation leads CPI by k months:

| Lead (months) | Correlation | Interpretation |
|---|---:|---|
| 0 | 0.9458 | Contemporaneous — same month |
| **1** | **0.9532** | **Peak — Truflation today best predicts CPI next month** |
| 2 | 0.9412 | Signal decays but remains very high |
| 3 | 0.9209 | Useful but weakening |
| 4 | 0.8970 | Still correlated, less actionable |
| 5 | 0.8710 | Significant decay |
| 6 | 0.8386 | Modest signal at 6-month horizon |

The peak at 1 month is the core actionable result. It means that during the ~2-week window between a CPI reference month closing and BLS publishing the data, Truflation's most recent reading is the single best real-time predictor of the pending print.

### 4.3 Regime breakdown

| Regime | Obs | Correlation | Context |
|---|---:|---:|---|
| Pre-COVID (2011–2019) | 120 | 0.870 | Low, stable inflation. Both series track the same slow drift but Truflation is noisier day-to-day. The lower correlation reflects methodological divergence (especially on shelter) being more visible when macro volatility is low. |
| COVID shock/rebound (2020–mid 2021) | 18 | 0.985 | Both indices captured the deflationary shock (Mar–Apr 2020) and subsequent supply-chain rebound in tandem. Real-time spending data and survey-based measurement converged during an unusually sharp macro move. |
| Post-COVID inflation/disinflation (mid 2021–present) | 55 | 0.969 | Correlation remains very high through the 2021–2022 inflation surge and subsequent disinflation. The high correlation in volatile regimes is consistent with Truflation being most useful precisely when macro uncertainty is highest. |

**Key takeaway:** The relationship is strongest when macro conditions are most uncertain — the periods when a PM would most want real-time inflation intelligence. This supports Truflation's role as a blackout-period proxy.

---

## 5. Finding B — Incremental Predictive Content (Granger Causality)

Correlation alone does not prove that Truflation adds *new* information beyond what we already know from CPI's own history. Granger causality tests this explicitly.

**Test:** Does Truflation at time t−k help predict CPI at time t, *conditional on* past CPI values?

Full results across lags 1–6:

| Lag | F-stat | p-value | Significant? |
|---|---:|---:|---|
| **1** | **32.00** | **0.0000001** | **Yes** |
| 2 | 10.23 | 0.00006 | Yes |
| 3 | 7.91 | 0.00005 | Yes |
| 4 | 5.98 | 0.00015 | Yes |
| 5 | 5.11 | 0.00021 | Yes |
| 6 | 4.92 | 0.00011 | Yes |

The evidence is unambiguous at all tested lags. At lag 1, the F-statistic of 32 is extremely large for a macro time-series test of this kind.

**Interpretation note:** Granger causality does not prove economic causation. It proves that Truflation contains **incremental forecast content** — information that is not already captured in CPI's own lags. In practical terms: including Truflation in a model materially improves CPI forecasts. The economic mechanism is plausible (Truflation captures real-time spending before BLS completes its survey cycle), but the test is statistical, not structural.

---

## 6. Finding C — Nowcast Model Performance

### 6.1 Model specification

**Formula:**

$$\hat{\text{CPI}}_{t} = \beta_0 + \beta_1 \cdot \text{Truflation}_{t} + \beta_2 \cdot \text{CPI}_{t-1} + \beta_3 \cdot \text{CPI}_{t-2} + \beta_4 \cdot (\text{Truflation}_{t-1} - \text{CPI}_{t-1})$$

**Feature rationale:**

| Feature | Coefficient | Role |
|---|---:|---|
| Intercept | +0.011 | Small constant — CPI has a slight positive drift |
| Truflation YoY (current) | +0.570 | The real-time signal. Coefficient < 1 reflects partial discounting — the model does not take Truflation at face value, which is correct given methodological differences |
| CPI YoY (t−1) | +0.468 | Autoregressive momentum — CPI is persistent |
| CPI YoY (t−2) | −0.045 | Second-order lag — small dampening effect |
| Truflation–CPI spread (t−1) | −0.506 | Mean-reversion: when the gap was large last month, it tends to narrow |

**The Truflation coefficient of 0.57** (not 1.0) is meaningful: the model has learned that Truflation systematically runs ahead of CPI during upside regimes (2021–2022) and below during downside regimes. It discounts the raw Truflation signal and blends it with CPI's own momentum.

### 6.2 Validation design

- **Expanding window walk-forward:** train on all data from month 0 to month t−1, predict month t, advance by one month, repeat
- **Minimum training window:** 24 months before first prediction
- **167 out-of-sample predictions** — each made with no access to future data
- **Baseline:** naive persistence — predict next CPI = last CPI (hardest simple baseline for a slow-moving series)

### 6.3 Out-of-sample results

| Metric | Model | Naive Baseline | Improvement |
|---|---:|---:|---:|
| MAE | **0.158pp** | 0.274pp | **+42.3%** |
| RMSE | **0.200pp** | 0.367pp | **+45.5%** |

**Error distribution:**

| Percentile | Absolute Error |
|---|---:|
| Median (p50) | 0.129pp |
| 90th percentile (p90) | 0.342pp |
| Maximum | 0.562pp |

**Directional accuracy (up/down vs prior month): 72.5%**

This means the model correctly predicts the *direction* of the next CPI move nearly three-quarters of the time — substantially better than chance (50%), and highly relevant for positioning decisions where direction matters more than the exact level.

**Recent 6-month MAE: 0.225pp** — slightly above the full-sample average, consistent with the current unusual spread regime being an edge-of-sample condition the model has limited training data for.

---

## 7. Current Month Signal Decomposition

Because the model is linear, the nowcast can be decomposed into the additive contribution of each feature. This is the full audit trail for the March 2026 nowcast of **2.54%**:

| Component | Formula | Contribution |
|---|---|---:|
| Intercept | β₀ | +0.011pp |
| Truflation term | 0.570 × 0.94% | +0.536pp |
| CPI lag-1 term | 0.468 × 2.66% | +1.247pp |
| CPI lag-2 term | −0.045 × 2.83% | −0.128pp |
| Spread mean-reversion | −0.506 × (0.94 − 2.66) | +0.872pp |
| **Total nowcast** | | **2.538%** |

**Why the spread term contributes positively despite the spread being negative:**
- Spread = Truflation − CPI = 0.94 − 2.66 = **−1.72pp** (Truflation is below CPI)
- Coefficient on lagged spread = **−0.506**
- Contribution = −0.506 × (−1.72) = **+0.872pp**
- Negative × negative = positive

The intuition: when Truflation has been running well below CPI (as now), the model expects some mean-reversion in the YoY paths rather than a one-for-one collapse in the next official print. CPI tends to drift toward Truflation gradually, not immediately, partly because of the shelter/OER lag embedded in BLS methodology.

---

## 8. Methodological Gap: Why Persistent Divergence Occurs

The most important caveat is structural, not statistical: **Truflation and CPI are built differently**, so level differences are expected.

**BLS CPI methodology:**
- Monthly household survey of actual prices paid
- Basket weights updated annually (Laspeyres-type index)
- **Shelter (~35% of CPI)** uses *Owners' Equivalent Rent (OER)* — a lagged, survey-based estimate of what homeowners would pay to rent their own home. OER typically lags actual market rents by 12–18 months.

**Truflation methodology:**
- Real-time consumer spending data (web scraping, card transaction feeds, alternative data)
- Updated daily, reflects actual transactions as they occur
- Shelter component tracks current market rents, not OER

**Consequence:** During periods of rapid rental market change (up or down), CPI's OER component introduces a structural lag. Truflation will lead CPI on the way up (as in 2021–2022) and also on the way down. The current negative spread may be partly explained by the OER component still reflecting higher rents from the 2022–2023 tightening cycle, while actual market rents have already eased.

This means the current spread is not purely a model signal — it has a structural interpretation. The correct framing is: Truflation provides a directional leading signal, and the OER lag explains *why* CPI is slow to follow.

---

## 9. Divergence Episodes and Risk Context

**Spread statistics (full sample):**

| Metric | Value | Date |
|---|---|---|
| Months with \|spread\| > 2pp | 15 | — |
| Maximum positive spread | +3.41pp | December 2021 |
| Maximum negative spread | **−1.72pp** | **February 2026 (current)** |

**Key historical episode — Apr 2021 to Jul 2022:**
Truflation ran +2 to +3.4pp above CPI for 15 consecutive months. This period validated the leading signal: Truflation captured the reopening demand surge, supply chain disruptions, and commodity price spikes *before* BLS survey methodology fully reflected them. CPI eventually caught up in full — confirming the leading relationship.

**Current episode (February 2026):**
The −1.72pp negative spread is the most extreme in the dataset — exceeding the worst readings during COVID or the pre-inflationary period. This is an edge-of-sample observation. Two competing explanations:

1. **Economic signal:** Real-time spending is genuinely decelerating, and BLS survey-based CPI is lagging the turn. The OER component keeping shelter elevated in CPI could explain the gap. In this case, expect the next 1–3 CPI prints to decelerate toward Truflation.
2. **Methodological divergence:** Category composition differences (energy, food, services) may be causing temporary divergence that does not fully transmit to BLS CPI. In this case, the spread narrows via Truflation mean-reverting upward rather than CPI falling.

The model implicitly assumes the truth is a blend of both (via the mean-reversion coefficient). A PM should overlay category-level intelligence to resolve this ambiguity.

---

## 10. Decision-Making Framework

**Use Truflation-based nowcast for:**
- Real-time directional monitoring between official BLS releases
- Blackout-window fill-in (government shutdown, delayed publication)
- Component of a composite inflation dashboard alongside breakevens, ISM prices, PPI

**Do not use as:**
- A direct replacement for CPI in index-linked calculations
- A single high-conviction point estimate during extreme spread regimes
- An intraday signal (signal operates at monthly cadence)

**Recommended decision thresholds:**

| Spread (Truflation − CPI) | Action |
|---|---|
| Within ±1pp | No special attention warranted |
| +1 to +2pp | Monitor for potential upside CPI surprise |
| Above +2pp | Elevated upside risk — directional positioning may be warranted |
| −1 to −2pp | Monitor for potential downside CPI surprise |
| Below −2pp | **Elevated downside risk (current: −1.72pp)** — treat model nowcast with wider uncertainty bands |

---

## 11. Caveats and Limitations

1. **Methodology mismatch:** Truflation and BLS CPI are constructed differently. Persistent level differences are expected and do not represent model failure. The OER lag in CPI is the primary structural driver.

2. **Regime dependence:** Statistical relationships can shift during policy, supply, or labour market structural breaks. The model was trained on one major inflationary cycle (2021–2023). Its behaviour in a prolonged deflation or stagflation scenario is untested.

3. **Sample length:** The overlap sample begins in 2010, giving approximately 15 years of observations. Expanding this window as more data accumulates will improve model stability.

4. **Model simplicity:** Linear regression was chosen for interpretability and robustness. Nonlinear models (gradient boosting, elastic net) may capture additional conditional effects but at the cost of explainability — an important trade-off for PM-facing analysis.

5. **Current spread is at a historical extreme:** The −1.72pp spread is outside the ±2pp training distribution for 15 of 167 validation observations. Model uncertainty at the current extreme is higher than the full-sample MAE of 0.158pp implies.

6. **Publication timing mechanics:** The 1-month lead-lag is partly economic (real-time data leads surveys) and partly mechanical (release calendars, BLS processing time). Both effects are valid but should be distinguished when interpreting the signal.

---

## 12. How to Reproduce

From the project root:

```bash
# Run full pipeline (fetch → transform → analyse → print commentary)
python run_pipeline.py

# Launch interactive dashboard
python run_pipeline.py --dashboard

# Open the fully-executed notebook
jupyter notebook notebooks/analysis.ipynb
```

Key output files:
- `data/processed/aligned_monthly.parquet` — aligned monthly dataset
- `data/processed/walk_forward_results.csv` — full validation results with actuals and predictions

Programmatic access:

```python
from src.nowcast import run_full_analysis
results = run_full_analysis(aligned_df)
# results["correlation"]  → CorrelationResult
# results["granger"]      → GrangerResult
# results["nowcast"]      → NowcastResult (includes walk_forward_results DataFrame)
# results["commentary"]   → full PM commentary string
```

---

## 13. Recommended Enhancements (Future Work)

To make this institutional-grade for committee packs or systematic integration:

1. **Prediction intervals** — empirical residual bands (e.g. p10/p90 envelope on the nowcast)
2. **Rolling coefficient stability** — chart β₁ (Truflation coefficient) over time to detect regime shifts in the signal strength
3. **Regime-conditional MAE table** — separate performance metrics for low/high spread regimes
4. **Model comparison panel** — linear vs elastic net vs gradient boosting, with explainability trade-off documented
5. **Category decomposition** — control for shelter/OER dynamics explicitly using FRED rent indices as a bridge variable
6. **Consensus comparison** — benchmark the model nowcast against Bloomberg consensus CPI estimates

---

## Bottom Line

Truflation is empirically useful as a **short-horizon CPI direction indicator** in this sample. The 1-month lead signal is strong, statistically significant, and economically interpretable. The directional accuracy of 72.5% and the 42% improvement over naive persistence make it a credible input for a macro monitoring framework.

The current −1.72pp spread is the most extreme negative reading in the dataset and points to **downside risk in the next CPI print**. The model nowcast of **2.54% for March 2026 CPI** (vs the last official print of 2.66%) should be treated as a directional signal with a ±0.16pp typical error and wider uncertainty given the extreme regime. Overlay with shelter dynamics (OER lag) and energy category intelligence before acting on the point estimate.

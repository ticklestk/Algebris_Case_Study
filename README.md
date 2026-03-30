# Algebris Data Analytics Case Study
## Truflation as a CPI Proxy: Data Pipeline, Dashboard & Nowcasting

---

## What This Project Does

A Portfolio Manager wants to know: *can Truflation's daily inflation data stand in for the official CPI when government data releases are delayed (e.g. during a US shutdown)?*

This project answers that in three parts:
1. **Data pipeline** — automatically fetches Truflation (daily) and official CPI from FRED (monthly), aligns them, and stores the results
2. **Dashboard** — interactive Plotly Dash app comparing both series with recession shading and a date-range slider
3. **Nowcasting model** — linear regression that uses today's Truflation reading to predict the *next* official CPI print, with walk-forward validation and error metrics vs a naive baseline

**Main deliverable:** `notebooks/analysis.ipynb` — fully executed Jupyter notebook with all charts, analysis, and a written Portfolio Manager commentary.

---

## Files — What's Used and What Isn't

```
algebris-case-study/
│
├── notebooks/
│   ├── analysis.ipynb        ✅ MAIN DELIVERABLE — open this in Jupyter
│   └── analysis.py           ⚠️  OLD VERSION — superseded by analysis.ipynb, ignore
│
├── src/                      ✅ All files here are active and used
│   ├── config.py             — API URLs, file paths, FRED key config (reads from .env)
│   ├── data_ingestion.py     — Fetches Truflation + FRED CPI + recession data
│   ├── transformations.py    — Cleans data, computes YoY %, aligns monthly
│   ├── nowcast.py            — Correlation, Granger causality, nowcasting model
│   ├── dashboard.py          — Plotly Dash interactive dashboard
│   ├── utils.py              — Retry logic, logging, Parquet/CSV helpers
│   └── __init__.py           — Package marker
│
├── tests/
│   └── test_transformations.py  ✅ Unit tests for the transformation layer
│
├── data/
│   ├── raw/                  — Raw API responses saved as Parquet (auto-created)
│   └── processed/            — Cleaned, aligned datasets (auto-created)
│
├── run_pipeline.py           ✅ CLI — run the pipeline, launch dashboard, or schedule
├── requirements.txt          ✅ Python dependencies
├── .env                      ✅ Your API keys (FRED key lives here)
├── .gitignore                ✅ Excludes .venv, data/, .env from git
│
├── .Rhistory                 ❌ NOT USED — can be deleted (R history file)
└── Algebris Data Analytics Case Study.docx  — Original brief from Algebris
```

---

## How to Run It

### 1. Setup (one time)

```bash
# From the project root:
cd algebris-case-study

# Activate the virtual environment (already set up)
source .venv/bin/activate

# Your .env already has the FRED API key — nothing to change
```

### 2. Open the Notebook (main deliverable)

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook is **fully pre-executed** — all outputs and charts are already embedded. You can just open and read it. If you want to re-run it from scratch:
- Click **Kernel → Restart & Run All** in Jupyter

### 3. Run the Data Pipeline (fetch fresh data)

This fetches the latest data from both APIs, runs the analysis, and prints the Portfolio Manager commentary:

```bash
python run_pipeline.py
```

Output saved to:
- `data/raw/truflation_raw.parquet` — raw Truflation daily data
- `data/raw/cpi_fred_raw.parquet` — raw FRED CPI data
- `data/processed/aligned_monthly.parquet` — aligned monthly dataset
- `data/processed/truflation_daily.parquet` — daily Truflation with YoY
- `data/processed/walk_forward_results.csv` — nowcast model CV results

### 4. Launch the Interactive Dashboard

```bash
python run_pipeline.py --dashboard
```

Then open **http://localhost:8050** in your browser.

The dashboard shows:
- CPI vs Truflation YoY % on the same axis with NBER recession shading
- Spread chart (Truflation − CPI divergence)
- Date-range slider for zooming
- Rolling 12-month correlation chart
- Summary stats cards (latest readings, correlation, MAE)

### 5. Run on a Daily Schedule (optional)

```bash
python run_pipeline.py --schedule
# Runs automatically every day at 08:00 UTC — press Ctrl+C to stop
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Data Sources

| Source | Series | Frequency | How fetched |
|---|---|---|---|
| [Truflation](https://truflation.com) | US CPI Inflation Index (YoY %) | Daily | REST API — no key needed, requires `User-Agent` header |
| [FRED](https://fred.stlouisfed.org) | CPIAUCSL — CPI-U All Items, SA | Monthly | `fredapi` Python library — free API key required |
| [FRED](https://fred.stlouisfed.org) | USREC — NBER Recession Indicator | Monthly | Same as above — used for recession shading on charts |

---

## Key Results (as of latest run)

| Metric | Value |
|---|---|
| Contemporaneous correlation | **0.945** |
| Best lead (Truflation ahead of CPI) | **1 month** (ρ = 0.953) |
| Granger causality p-value | **< 0.0001** (significant) |
| Nowcast model MAE | **0.158 pp** |
| Naive baseline MAE | **0.274 pp** |
| Improvement over baseline | **+42%** |
| March 2026 CPI nowcast | **2.54% YoY** |

---

## How the Pipeline Works

```
Truflation API ──┐
                 ├──► data_ingestion.py ──► transformations.py ──► nowcast.py ──► commentary
FRED API ────────┘         │                      │                    │
                       raw Parquet           processed Parquet     walk_forward.csv
                                                   │
                                             dashboard.py ──► http://localhost:8050
```

**Step by step:**

1. `data_ingestion.py` fetches both APIs with retry logic and saves raw Parquet files
2. `transformations.py` computes YoY % for CPI (12-month lag on index level), forward-fills Truflation gaps, resamples Truflation to monthly, and aligns both on month-start dates
3. `nowcast.py` runs correlation analysis, Granger causality test, and a walk-forward linear regression model — all using only data available at prediction time (no lookahead)
4. `dashboard.py` loads the processed data and serves the Dash app

---

## Design Decisions

**FRED over BLS direct API** — FRED provides a cleaner, more consistent API with useful auxiliary series (recession indicator, inflation expectations). The BLS API has stricter rate limits and less convenient formatting.

**Pydantic validation on API responses** — Both ingestion functions validate the raw API response through typed Pydantic models. This catches schema changes early rather than silently producing bad data — important for a pipeline that may run unattended.

**Linear regression for the nowcast model** — Deliberately chosen over black-box alternatives (gradient boosting, neural nets) for full explainability. The Portfolio Manager can see exactly what the model is doing from the coefficient table. The features are: current Truflation YoY, CPI lag-1, CPI lag-2, and the lagged Truflation−CPI spread.

**Walk-forward expanding-window CV** — The only correct way to validate a time series forecasting model. Train on [0…t−1], predict t, expand. No data leakage by construction. Naive persistence (predict next CPI = last CPI) is used as the hard baseline to beat.

**Parquet persistence between stages** — Each pipeline stage saves its output to Parquet. If the API goes down or you want to re-run just the analysis without re-fetching, you can. The dashboard also loads from Parquet rather than hitting the API every time.

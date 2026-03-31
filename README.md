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

## Project Structure

```
algebris-case-study/
│
├── notebooks/
│   └── analysis.ipynb        # Main deliverable — fully executed, open this in Jupyter
│
├── src/
│   ├── config.py             # API URLs, file paths, FRED key (reads from .env)
│   ├── data_ingestion.py     # Fetches Truflation + FRED CPI + recession data
│   ├── transformations.py    # Cleans data, computes YoY %, aligns monthly
│   ├── nowcast.py            # Correlation, Granger causality, nowcasting model
│   ├── dashboard.py          # Plotly Dash interactive dashboard
│   ├── utils.py              # Retry logic, logging, Parquet/CSV helpers
│   └── __init__.py
│
├── tests/
│   └── test_transformations.py  # Unit tests for the transformation layer
│
├── run_pipeline.py           # CLI entry point — pipeline, dashboard, scheduler
├── requirements.txt          # Python dependencies
├── .env.example              # Template for required environment variables
└── Algebris Data Analytics Case Study.docx  # Original brief
```

`data/` is created automatically when the pipeline runs and is not tracked in git.

---

## Setup

### Prerequisites
- Python 3.10+
- A free FRED API key — get one at https://fred.stlouisfed.org/docs/api/api_key.html (takes ~1 minute)

### Install

```bash
# Clone and enter the repo
git clone https://github.com/ticklestk/Algebris_Case_Study.git
cd Algebris_Case_Study

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Open .env and replace 'your_fred_api_key_here' with your actual FRED key
```

---

## How to Run It

### View the notebook (main deliverable)

The notebook is fully pre-executed — all outputs and charts are already embedded. Just open it:

```bash
jupyter notebook notebooks/analysis.ipynb
```

To re-run it from scratch: **Kernel → Restart & Run All** in Jupyter.

### Run the data pipeline (fetch fresh data)

Fetches the latest data from both APIs, runs the full analysis, and prints the Portfolio Manager commentary:

```bash
python run_pipeline.py
```

Saves output to:
- `data/raw/truflation_raw.parquet`
- `data/raw/cpi_fred_raw.parquet`
- `data/processed/aligned_monthly.parquet`
- `data/processed/truflation_daily.parquet`
- `data/processed/walk_forward_results.csv`

### Live demo

A hosted version is available at **https://algebris-case-study.onrender.com**

> **Note:** The app is hosted on Render's free tier, which spins down after ~15 minutes of inactivity. The first visit after a period of inactivity may take **30–60 seconds** to load — this is normal. Subsequent requests are instant.

### Launch the interactive dashboard locally

```bash
python run_pipeline.py --dashboard
```

Open **http://localhost:8050** in your browser. The dashboard includes:
- CPI vs Truflation YoY % on the same axis with NBER recession shading
- Spread chart (Truflation − CPI divergence)
- Date-range slider for zooming
- Rolling 12-month correlation chart
- Summary statistics cards

### Run on a daily schedule

```bash
python run_pipeline.py --schedule
# Runs automatically every day at 08:00 UTC — Ctrl+C to stop
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Data Sources

| Source | Series | Frequency | Notes |
|---|---|---|---|
| [Truflation](https://truflation.com) | US CPI Inflation Index (YoY %) | Daily | No API key needed |
| [FRED](https://fred.stlouisfed.org) | CPIAUCSL — CPI-U All Items, SA | Monthly | Free API key required |
| [FRED](https://fred.stlouisfed.org) | USREC — NBER Recession Indicator | Monthly | Used for recession shading |

---

## Key Results

Truflation and official CPI YoY are strongly correlated (ρ = 0.945) with a **1-month lead** — today's Truflation reading correlates most strongly with CPI published next month. Granger causality is confirmed (p < 0.0001), meaning Truflation adds predictive information beyond CPI's own history. A linear nowcast model improves **+42% over a naive baseline** in walk-forward out-of-sample testing, with directional accuracy of 72.5%.

Current signal (as of February 2026): Truflation at 0.94% is **1.72pp below** CPI at 2.66% — the widest negative spread on record — pointing to downside risk on the next print. Model nowcast for March 2026: **2.54% YoY**.

| Metric | Value |
|---|---|
| Contemporaneous correlation | **0.945** |
| Best lead (Truflation ahead of CPI) | **1 month** (ρ = 0.953) |
| Granger causality p-value | **< 0.0001** (significant) |
| Nowcast model MAE | **0.158 pp** |
| Naive baseline MAE | **0.274 pp** |
| Improvement over baseline | **+42%** |
| Directional accuracy | **72.5%** |
| March 2026 CPI nowcast | **2.54% YoY** |

For full analysis including regime correlation breakdown, signal decomposition, historical divergence episodes, and error percentiles — see [`FINDINGS_DETAILED.md`](FINDINGS_DETAILED.md). The written PM commentary is in Section 4 of [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) and rendered live on the dashboard.

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

1. `data_ingestion.py` — fetches both APIs with retry/backoff, validates responses with Pydantic, saves raw Parquet
2. `transformations.py` — computes YoY % for CPI (12-month lag), forward-fills Truflation gaps, resamples to monthly, aligns on month-start dates
3. `nowcast.py` — correlation analysis, Granger causality test, walk-forward linear regression (no lookahead bias)
4. `dashboard.py` — loads processed Parquet and serves the Dash app

---

## Design Decisions

**FRED over BLS direct API** — FRED provides a cleaner API with consistent formatting and includes useful auxiliary series (recession indicator). The BLS API has stricter rate limits and less convenient date formatting.

**Pydantic validation on API responses** — Both ingestion functions validate the raw response through typed Pydantic models. This catches schema changes early rather than silently producing bad data — critical for a pipeline that may run unattended.

**Linear regression for the nowcast model** — Deliberately chosen over black-box alternatives for full explainability. The Portfolio Manager can read the coefficient table and understand exactly what is driving the prediction. Features: current Truflation YoY, CPI lag-1, CPI lag-2, lagged Truflation−CPI spread.

**Walk-forward expanding-window CV** — The only correct validation approach for a time series forecasting model. Train on [0…t−1], predict t, expand. No data leakage by construction. Naive persistence (predict next CPI = last CPI) is used as the baseline.

**Parquet persistence between stages** — Each stage saves output to Parquet. If an API is down or you want to re-run analysis without re-fetching, you can. The dashboard loads from Parquet rather than hitting APIs on every page load.

---

## AI Tools Disclosure

Claude (Anthropic) was used as a coding assistant throughout this project:

| Area | How AI was used |
|---|---|
| Project scaffolding | Suggested modular `src/` package structure, `pydantic-settings` config pattern, retry decorator boilerplate |
| API integration | Helped debug Truflation API (403 header fix, new response format parsing, SSL cert issue on macOS) |
| Validation logic | Reviewed walk-forward CV for lookahead bias |
| Documentation | Assisted with docstrings and notebook narrative |

**What was not delegated to AI:**
- Feature selection for the nowcasting model
- Choice of linear regression for explainability over black-box alternatives
- Analytical interpretation of results and PM commentary framing
- Decision to use Granger causality as the formal predictive-power test
- All data quality and alignment decisions

All code was reviewed, tested, and adapted to the specific data characteristics encountered. Full disclosure is also included in Section 6 of `notebooks/analysis.ipynb`.

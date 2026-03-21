# Cash Pot — Lottery prediction (ML demo)

Flask web app for exploring **historical Cash Pot–style draw data** with **Random Forest** and **LSTM (TensorFlow)** models, plus frequency/pattern heuristics. Draw scheduling follows the **Jamaica** timetable (six daily draws, excluding Christmas and Good Friday).

> **Disclaimer:** This project is for **learning and experimentation** only. Past draws do not predict future outcomes. Do not use this as financial or gambling advice.

## Features

- **Dashboard** — Stats, recent predictions, accuracy summary, optional startup training banner, model health / monitor hooks
- **Predictions** — Multiple methods (e.g. LSTM, Random Forest, frequency, patterns, ensemble/auto with governance hooks)
- **Data** — CSV upload, single-draw entry, edit/delete draws
- **History** — Paginated draw history and prediction history; relink predictions to draws
- **Retraining** — Async `/retrain` with `/api/training_status` for UI feedback
- **Backtesting & benchmarks** — Walk-forward backtest, multi-seed evaluation, benchmark baseline/history, profile presets (`quick` / `standard` / `deep`)
- **APIs** — JSON endpoints for stats, draws, backtest, benchmark evaluate/history, monitor health, model/training status

## Tech stack

| Layer | Choice |
|--------|--------|
| Web | Flask 2.x |
| Data | SQLite (via `DatabaseManager` in `models.py`) |
| ML | scikit-learn (Random Forest), TensorFlow 2.13 (LSTM softmax + calibration) |
| Deploy-friendly | `gunicorn` in `requirements.txt` |

## Requirements

- **Python 3.10+** recommended (TensorFlow 2.13 is picky on version; use a version that matches your platform)
- System dependencies as required by TensorFlow on your OS

## Setup

```bash
cd "Gambling App"   # or your clone path
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_STARTUP_TRAINING` | `1` | If `0`, skip training models when the app starts (faster dev; use **Retrain** in UI/API instead) |

**Production:** Set a real `Flask` `secret_key` in `app.py` or via env/config — do not ship the placeholder.

### Database & models

- SQLite file: **`lottery_data.db`** (created when you first run the app). Listed in `.gitignore` — **not** committed by default.
- Trained artifacts may live under **`trained_models/`** (add to `.gitignore` if you do not want them in Git).

## Run locally

```bash
source .venv/bin/activate
python3 app.py
```

Open the URL shown in the terminal (often `http://127.0.0.1:5000`).

**Production-style (example):**

```bash
gunicorn -w 2 -b 0.0.0.0:8080 app:app
```

## Project layout (high level)

| Path | Role |
|------|------|
| `app.py` | Routes, training thread state, benchmark profiles |
| `models.py` | SQLite schema, draws, predictions, benchmarks |
| `predictor.py` | `LotteryPredictor` — methods, ensemble, backtests |
| `ml_trainer.py` | LSTM training/inference, RF helper |
| `schedule.py` | Jamaica draw times & blackout dates |
| `templates/` | HTML pages |
| `static/` | CSS/JS |

## API quick reference

| Method | Path | Notes |
|--------|------|--------|
| GET | `/` | Dashboard |
| POST | `/predict` | JSON prediction |
| GET/POST | `/upload`, `/add_single`, … | Data management |
| GET | `/history`, `/prediction-history` | Paginated views |
| POST | `/retrain` | Trigger retrain (background) |
| GET | `/api/training_status`, `/api/model_status` | Training / model state |
| GET | `/api/backtest`, `/api/backtest_multi` | Backtest JSON |
| GET/POST | `/api/benchmark/*` | Baseline, evaluate, history |
| GET | `/api/monitor/health` | Lightweight health JSON |

## License

Specify your license here (e.g. MIT) if you want the repo to be open source.

## Author

**loukieluke** — [MachineLearning](https://github.com/loukieluke/MachineLearning)

# Satellite Health Monitoring with Apache Airflow

An end-to-end **Airflow pipeline** for synthetic satellite telemetry: generate data, preprocess, engineer features, train an anomaly-detection model (Isolation Forest), score data, and produce health reports—all orchestrated by **Apache Airflow**.

---

## 📋 Overview

| Component | Description |
|-----------|-------------|
| **Orchestration** | Apache Airflow 2.x (Docker Compose) |
| **ML model** | Isolation Forest (scikit-learn) for unsupervised anomaly detection |
| **Data** | Synthetic telemetry: temperature, voltage, current, angular velocity, solar power |
| **Output** | Health reports + scored CSV with predicted anomalies |

---

## 🏗️ Project Structure

```
Airflow/
├── docker-compose.yml    # Airflow + Postgres + Redis (LocalExecutor)
├── .env.example          # Template for .env
├── README.md
└── dags/
    ├── satellite_health_ml.py      # Full ML pipeline DAG
    ├── satellite_health_simple.py   # Simple learning DAG (no ML)
    ├── data/                        # CSVs and reports (mounted into containers)
    ├── model/                       # Saved model (anomaly_model.pkl)
    └── src/                         # Reusable Python modules
        ├── generate_data.py         # Synthetic telemetry generation
        ├── preprocess.py            # Clean & resample to hourly
        ├── features.py              # Rolling stats, deltas, power
        ├── train.py                 # Isolation Forest training
        └── score.py                 # Anomaly scoring
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker**
- (Optional) Python 3.8+ for local scripts

### 1. Clone and Configure

```bash
git clone https://github.com/prashanthjaganathan/MLOps_Sandbox.git
cd Airflow
cp .env.example .env
```

### 2. Initialize Airflow (one-time)

```bash
docker-compose up airflow-init
# Wait until you see "airflow-init exited with code 0"
```

### 3. Start services

```bash
docker-compose up -d
```

### 4. Open the UI

- **URL:** [http://localhost:8080](http://localhost:8080)
- **Login:** `airflow` / `airflow` (or values from `.env`)

### 5. Run the pipeline

1. Find **`satellite_health_ml_pipeline`** in the UI.
2. Turn the DAG **On** (toggle).
3. Click **Trigger DAG** (play button).

---

## 📊 Pipeline: `satellite_health_ml_pipeline`

| Step | Task | Description |
|------|------|-------------|
| 1 | **generate_data** | Generate 90 days of synthetic telemetry → `dags/data/telemetry.csv` |
| 2 | **ingest_telemetry** | Load and validate raw CSV; return path via XCom |
| 3 | **preprocess_data** | Clean, resample to hourly → `telemetry_cleaned.csv` |
| 4 | **engineer_features** | Rolling means/std, deltas, power → `telemetry_features.csv` |
| 5 | **train_model** | Train Isolation Forest → `dags/model/anomaly_model.pkl` |
| 6 | **score_telemetry** | Predict anomalies → `telemetry_scored.csv` |
| 7 | **generate_health_report** | Write summary + metrics → `ml_health_report_*.txt` |

Schedule: **daily** (`@daily`). No backfill (`catchup=False`).

---

## 📁 Key Artifacts

| File | Produced by | Description |
|------|-------------|-------------|
| `dags/data/telemetry.csv` | generate_data | Raw synthetic telemetry (90 days, 1 sample/min) |
| `dags/data/telemetry_cleaned.csv` | preprocess_data | Hourly resampled data |
| `dags/data/telemetry_features.csv` | engineer_features | With rolling stats and power |
| `dags/model/anomaly_model.pkl` | train_model | Trained Isolation Forest |
| `dags/data/telemetry_scored.csv` | score_telemetry | Predictions + anomaly scores |
| `dags/data/ml_health_report_*.txt` | generate_health_report | Human-readable report |

---

## ⚙️ Configuration

### Environment (`.env`)

- **AIRFLOW_UID** – User ID for file permissions (default `50000`; on Linux, use `id -u`).
- **AIRFLOW_PROJ_DIR** – Project root for volume paths (default `.`).
- **_AIRFLOW_WWW_USER_USERNAME / _AIRFLOW_WWW_USER_PASSWORD** – Web UI login.
- **_PIP_ADDITIONAL_REQUIREMENTS** – Extra packages in containers (e.g. `scikit-learn pandas numpy`).

### Executor

The included `docker-compose.yml` is set up for **LocalExecutor**. For Celery/Redis, change the executor and enable worker/redis services in the compose file.

---

## 🧪 Other DAG: `satellite_health_simple`

A simpler DAG for learning: load telemetry → validate → basic stats + 3-sigma anomaly count → text report. No ML. Useful for understanding tasks, XCom, and the UI.

---

## 🛠️ Useful Commands

```bash
# Restart after changing .env or docker-compose
docker-compose down && docker-compose up -d

# View logs
docker-compose logs -f airflow-scheduler

# Run data generator manually (inside container)
docker-compose exec airflow-webserver python /opt/airflow/dags/src/generate_data.py
```

---

## 📚 References

- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Airflow Docker Compose](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose.html)
- [Isolation Forest (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

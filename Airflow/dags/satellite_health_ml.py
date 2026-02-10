"""
Complete Satellite Health Monitoring ML Pipeline

Pipeline order:
1. Generate data   – create/append telemetry.csv
2. Ingest          – load and validate raw data
3. Preprocess      – clean and resample
4. Engineer features
5. Train model     – Isolation Forest
6. Score telemetry
7. Generate health report
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow/dags/src')
from preprocess import clean_and_resample
from features import engineer_features
from train import train_anomaly_model
from score import score_telemetry
from generate_data import run_generate_and_save

# ============================================
# DEFAULT ARGUMENTS
# ============================================
default_args = {
    'owner': 'satellite_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# ============================================
# DAG DEFINITION
# ============================================
dag = DAG(
    'satellite_health_ml_pipeline',
    default_args=default_args,
    description='ML-based satellite health monitoring with Isolation Forest',
    schedule_interval='@daily',
    catchup=False,
    tags=['satellite', 'ml', 'production'],
)

DATA_PATH = '/opt/airflow/dags/data/telemetry.csv'

# ============================================
# TASK 1: GENERATE DATA
# ============================================

generate_data_task = PythonOperator(
    task_id='generate_data',
    python_callable=run_generate_and_save,
    op_kwargs={
        'output_path': DATA_PATH
    },
    dag=dag,
)

# ============================================
# TASK 2: INGEST TELEMETRY
# ============================================
@task(dag=dag)
def ingest_telemetry():
    """Load and validate raw telemetry file; return path for downstream tasks."""
    print("="*60)
    print("📡 INGESTION: Loading and validating telemetry")
    print("="*60)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Telemetry file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Loaded {len(df):,} samples")
    print(f"📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return DATA_PATH

# ============================================
# TASK 3: PREPROCESS
# ============================================
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=clean_and_resample,
    op_kwargs={
        'input_path': '{{ ti.xcom_pull(task_ids="ingest_telemetry") }}',
        'output_path': '/opt/airflow/dags/data/telemetry_cleaned.csv'
    },
    dag=dag,
)

# ============================================
# TASK 4: ENGINEER FEATURES
# ============================================
feature_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    op_kwargs={
        'input_path': '/opt/airflow/dags/data/telemetry_cleaned.csv',
        'output_path': '/opt/airflow/dags/data/telemetry_features.csv'
    },
    dag=dag,
)

# ============================================
# TASK 5: TRAIN MODEL
# ============================================
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_anomaly_model,
    op_kwargs={
        'input_path': '/opt/airflow/dags/data/telemetry_features.csv',
        'model_path': '/opt/airflow/dags/model/anomaly_model.pkl'
    },
    dag=dag,
)

# ============================================
# TASK 6: SCORE TELEMETRY
# ============================================
score_task = PythonOperator(
    task_id='score_telemetry',
    python_callable=score_telemetry,
    op_kwargs={
        'input_path': '/opt/airflow/dags/data/telemetry_features.csv',
        'model_path': '/opt/airflow/dags/model/anomaly_model.pkl',
        'output_path': '/opt/airflow/dags/data/telemetry_scored.csv'
    },
    dag=dag,
)

# ============================================
# TASK 7: GENERATE HEALTH REPORT
# ============================================
@task(dag=dag)
def generate_health_report():
    """Generate final health report from scored data."""
    print("="*60)
    print("📊 REPORTING: Generating health report")
    print("="*60)
    df = pd.read_csv('/opt/airflow/dags/data/telemetry_scored.csv')
    total_samples = len(df)
    total_anomalies = df['is_predicted_anomaly'].sum()
    anomaly_rate = (total_anomalies / total_samples) * 100

    if 'is_anomaly' in df.columns:
        actual_anomaly_label = (df['is_anomaly'] > 0)
        actual_anomalies = actual_anomaly_label.sum()
        true_positives = (actual_anomaly_label & (df['is_predicted_anomaly'] == 1)).sum()
        false_positives = (~actual_anomaly_label & (df['is_predicted_anomaly'] == 1)).sum()
        false_negatives = (actual_anomaly_label & (df['is_predicted_anomaly'] == 0)).sum()
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    else:
        actual_anomalies = true_positives = false_positives = false_negatives = precision = recall = None

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║      🛰️  SATELLITE HEALTH MONITORING REPORT (ML)            ║
╚══════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: Isolation Forest

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 DATASET SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples: {total_samples:,}
Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 SYSTEM AVERAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature:        {df['temperature'].mean():.2f}°C
Voltage:            {df['voltage'].mean():.2f}V
Current:            {df['current'].mean():.2f}A
Angular Velocity:   {df['angular_velocity'].mean():.4f} deg/s
Solar Power:        {df['solar_power'].mean():.2f}W

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 ML ANOMALY DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Anomalies: {total_anomalies:,} ({anomaly_rate:.2f}%)
Predicted Normal:    {total_samples - total_anomalies:,}
"""
    if actual_anomalies is not None:
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Actual Anomalies:    {actual_anomalies:,}
True Positives:      {true_positives:,}
False Positives:     {false_positives:,}
False Negatives:     {false_negatives:,}
Precision:           {precision:.2%}
Recall:              {recall:.2%}
"""
    status = "✅ NOMINAL" if anomaly_rate < 1 else "⚠️  CAUTION" if anomaly_rate < 5 else "🚨 ALERT"
    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 HEALTH STATUS: {status}
═══════════════════════════════════════════════════════════════
"""
    print(report)
    report_path = f"/opt/airflow/dags/data/ml_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Report saved to {report_path}")
    return report_path

# ============================================
# DEPENDENCIES: generate_data → ingest → preprocess → … → report
# ============================================
ingest = ingest_telemetry()
generate_data_task >> ingest >> preprocess_task >> feature_task >> train_task >> score_task >> generate_health_report()
"""
Simple Satellite Health Monitoring DAG - Learning Version

This DAG demonstrates basic Airflow concepts:
- Loading data
- Validating data quality
- Analyzing health metrics
- Generating reports
"""

from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
import pandas as pd

# ============================================
# DEFAULT ARGUMENTS
# ============================================
# These settings apply to ALL tasks in this DAG
default_args = {
    'owner': 'satellite_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ============================================
# DAG DEFINITION
# ============================================
dag = DAG(
    'satellite_health_simple',
    default_args=default_args,
    description='Simple satellite telemetry health check',
    schedule_interval='@daily',  # Run once per day
    catchup=False,               # Don't backfill past runs
    tags=['satellite', 'learning'],
)

# ============================================
# TASK 1: LOAD TELEMETRY DATA
# ============================================
@task(dag=dag)
def load_telemetry():
    """
    Load telemetry data from CSV file
    
    Returns:
        str: Path to the CSV file
    """
    print("="*60)
    print("TASK 1: Loading telemetry data...")
    print("="*60)
    
    file_path = '/opt/airflow/dags/data/telemetry.csv'
    
    # Load the data to check it exists and is valid
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df):,} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"File size: {len(df) * len(df.columns)} data points")
    
    # Return the file path (NOT the dataframe!)
    # Why? Because XCom stores data in database, and DataFrames are too big!
    return file_path

# ============================================
# TASK 2: VALIDATE DATA QUALITY
# ============================================
@task(dag=dag)
def validate_data(data_path: str):
    """
    Validate data quality and check for issues
    
    Args:
        data_path: Path to the CSV file (from previous task)
    
    Returns:
        str: Path to the validated CSV file
    """
    print("="*60)
    print("TASK 2: Validating data quality...")
    print("="*60)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Check 1: Required columns exist
    required_cols = ['timestamp', 'temperature', 'voltage', 'current', 'angular_velocity', 'solar_power']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"All required columns present: {required_cols}")
    
    # Check 2: No null values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"Warning: Found {null_count} null values")
        # In production, you might want to raise an error here
    else:
        print(f"No null values found")
    
    # Check 3: Data ranges are reasonable
    checks = []
    
    if df['temperature'].min() < -50 or df['temperature'].max() > 100:
        checks.append(f"Warning: Temperature out of expected range: {df['temperature'].min():.1f}C to {df['temperature'].max():.1f}C")
    else:
        checks.append(f"Temperature in valid range: {df['temperature'].min():.1f}C to {df['temperature'].max():.1f}C")
    
    if df['voltage'].min() < 20 or df['voltage'].max() > 35:
        checks.append(f"Warning: Voltage out of expected range: {df['voltage'].min():.1f}V to {df['voltage'].max():.1f}V")
    else:
        checks.append(f"Voltage in valid range: {df['voltage'].min():.1f}V to {df['voltage'].max():.1f}V")
    
    for check in checks:
        print(check)
    
    # Check 4: Data is sorted by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].is_monotonic_increasing:
        print("Data is properly sorted by timestamp")
    else:
        print("Warning: Data is not sorted by timestamp")
    
    print("\nValidation passed!")
    return data_path

# ============================================
# TASK 3: ANALYZE SATELLITE HEALTH
# ============================================
@task(dag=dag)
def analyze_health(data_path: str):
    """
    Analyze satellite health metrics and detect anomalies
    
    Args:
        data_path: Path to the validated CSV file
    
    Returns:
        dict: Health statistics
    """
    print("="*60)
    print("TASK 3: Analyzing satellite health...")
    print("="*60)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Calculate basic statistics
    stats = {
        'total_samples': len(df),
        'avg_temperature': df['temperature'].mean(),
        'avg_voltage': df['voltage'].mean(),
        'avg_current': df['current'].mean(),
        'avg_angular_velocity': df['angular_velocity'].mean(),
        'avg_solar_power': df['solar_power'].mean(),
    }
    
    print("Health Statistics:")
    print(f"  Total Samples: {stats['total_samples']:,}")
    print(f"  Avg Temperature: {stats['avg_temperature']:.2f}C")
    print(f"  Avg Voltage: {stats['avg_voltage']:.2f}V")
    print(f"  Avg Current: {stats['avg_current']:.2f}A")
    print(f"  Avg Angular Velocity: {stats['avg_angular_velocity']:.4f} deg/s")
    print(f"  Avg Solar Power: {stats['avg_solar_power']:.2f}W")
    
    # Simple anomaly detection: 3 standard deviations from mean
    print("\nAnomaly Detection (3-sigma rule):")
    
    temp_mean = df['temperature'].mean()
    temp_std = df['temperature'].std()
    temp_anomalies = df[abs(df['temperature'] - temp_mean) > 3 * temp_std]
    
    voltage_mean = df['voltage'].mean()
    voltage_std = df['voltage'].std()
    voltage_anomalies = df[abs(df['voltage'] - voltage_mean) > 3 * voltage_std]
    
    print(f"  Temperature anomalies: {len(temp_anomalies)} ({len(temp_anomalies)/len(df)*100:.2f}%)")
    print(f"  Voltage anomalies: {len(voltage_anomalies)} ({len(voltage_anomalies)/len(df)*100:.2f}%)")
    
    # Add anomaly counts to stats
    stats['temp_anomalies'] = len(temp_anomalies)
    stats['voltage_anomalies'] = len(voltage_anomalies)
    
    print("\nAnalysis complete!")
    return stats

# ============================================
# TASK 4: GENERATE HEALTH REPORT
# ============================================
@task(dag=dag)
def generate_report(stats: dict):
    """
    Generate a human-readable health report
    
    Args:
        stats: Health statistics from previous task
    
    Returns:
        str: Path to the generated report file
    """
    print("="*60)
    print("TASK 4: Generating health report...")
    print("="*60)
    
    # Create report content
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         SATELLITE HEALTH MONITORING REPORT                   ║
╚══════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples Analyzed: {stats['total_samples']:,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM AVERAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature:        {stats['avg_temperature']:.2f}C
Voltage:            {stats['avg_voltage']:.2f}V
Current:            {stats['avg_current']:.2f}A
Angular Velocity:   {stats['avg_angular_velocity']:.4f} deg/s
Solar Power:        {stats['avg_solar_power']:.2f}W

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANOMALY DETECTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature Anomalies: {stats['temp_anomalies']} ({stats['temp_anomalies']/stats['total_samples']*100:.2f}%)
Voltage Anomalies:     {stats['voltage_anomalies']} ({stats['voltage_anomalies']/stats['total_samples']*100:.2f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEALTH STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # Determine overall health status
    total_anomalies = stats['temp_anomalies'] + stats['voltage_anomalies']
    anomaly_rate = (total_anomalies / stats['total_samples']) * 100
    
    if anomaly_rate < 1:
        status = "NOMINAL - All systems operating normally"
    elif anomaly_rate < 5:
        status = "CAUTION - Minor anomalies detected"
    else:
        status = "ALERT - Significant anomalies require attention"
    
    report += f"Status: {status}\n"
    report += f"Overall Anomaly Rate: {anomaly_rate:.2f}%\n"
    report += "\n" + "═"*64 + "\n"
    
    # Print report to logs
    print(report)
    
    # Save report to file
    report_filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = f"/opt/airflow/dags/data/{report_filename}"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {report_path}")
    return report_path

# ============================================
# DEFINE TASK DEPENDENCIES
# ============================================
# This is where we define the order tasks run in!

# Method 1: Implicit dependencies (recommended for TaskFlow API)
data_path = load_telemetry()              # Step 1: Load data
validated_path = validate_data(data_path) # Step 2: Validate (waits for step 1)
health_stats = analyze_health(validated_path)  # Step 3: Analyze (waits for step 2)
report_path = generate_report(health_stats)    # Step 4: Report (waits for step 3)

# The above creates this flow:
# load_telemetry >> validate_data >> analyze_health >> generate_report
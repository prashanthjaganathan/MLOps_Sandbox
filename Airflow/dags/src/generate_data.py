"""
Generate synthetic satellite telemetry data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_telemetry_data(days=90, samples_per_day=1440):
    """
    Generate synthetic satellite telemetry data
    
    Parameters:
    - days: Number of days of data (default: 90 days = 3 months)
    - samples_per_day: Data points per day (default: 1440 = 1 per minute)
    
    Returns:
    - DataFrame with telemetry data
    """
    
    # Calculate total samples
    # Example: 90 days × 1440 samples/day = 129,600 total samples
    total_samples = days * samples_per_day
    
    print(f"Generating {total_samples:,} telemetry samples...")
    print(f"Duration: {days} days")
    print(f"Frequency: 1 sample per minute")
    
    # ============================================
    # STEP 1: Generate Timestamps
    # ============================================
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i) for i in range(total_samples)]
    
    print(f"Generated timestamps from {timestamps[0]} to {timestamps[-1]}")
    
    # ============================================
    # STEP 2: Generate Normal Operating Data
    # ============================================
    np.random.seed(42)  # For reproducibility
    
    # Temperature (Celsius) - Normal distribution around 20°C
    temp_mean = 20.0
    temp_std = 2.0
    temperature = np.random.normal(temp_mean, temp_std, total_samples)
    
    # Voltage (Volts) - Battery voltage around 28V
    voltage_mean = 28.0
    voltage_std = 0.5
    voltage = np.random.normal(voltage_mean, voltage_std, total_samples)
    
    # Current (Amps) - Power draw around 5A
    current_mean = 5.0
    current_std = 0.3
    current = np.random.normal(current_mean, current_std, total_samples)
    
    # Angular velocity (deg/sec) - Slow rotation
    angular_vel_mean = 0.1
    angular_vel_std = 0.02
    angular_velocity = np.random.normal(angular_vel_mean, angular_vel_std, total_samples)
    
    # Solar panel output (Watts)
    solar_power_mean = 140.0
    solar_power_std = 10.0
    solar_power = np.random.normal(solar_power_mean, solar_power_std, total_samples)
    
    print("Generated normal operating data")
    
    # ============================================
    # STEP 3: Create DataFrame
    # ============================================
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'voltage': voltage,
        'current': current,
        'angular_velocity': angular_velocity,
        'solar_power': solar_power
    })
    
    # ============================================
    # STEP 4: Inject Anomalies (5% of data)
    # ============================================
    num_anomalies = int(total_samples * 0.05)
    anomaly_indices = np.random.choice(total_samples, size=num_anomalies, replace=False)
    
    print(f"Injecting {num_anomalies:,} anomalies ({(num_anomalies/total_samples)*100:.1f}%)...")
    
    # Create different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.choice([
            'temp_spike',      # Overheating
            'voltage_drop',    # Battery dying
            'power_loss',      # Solar panel failure
            'spin'             # Uncontrolled rotation
        ])
        
        if anomaly_type == 'temp_spike':
            df.loc[idx, 'temperature'] = np.random.uniform(35, 45)  # Way too hot!
        elif anomaly_type == 'voltage_drop':
            df.loc[idx, 'voltage'] = np.random.uniform(20, 24)      # Low battery!
        elif anomaly_type == 'power_loss':
            df.loc[idx, 'solar_power'] = np.random.uniform(20, 60)  # Panel failure!
        elif anomaly_type == 'spin':
            df.loc[idx, 'angular_velocity'] = np.random.uniform(0.5, 1.0)  # Tumbling!
    
    # Mark anomalies
    df['is_anomaly'] = 0
    df.loc[anomaly_indices, 'is_anomaly'] = 1
    
    print("Anomalies injected successfully")
    
    return df

def run_generate_and_save(output_path='/opt/airflow/dags/data/telemetry.csv'):
    """
    Generate 90 days of telemetry and write to CSV. Used by the DAG's generate_data task.
    Returns output_path.
    """
    print("="*60)
    print("📦 GENERATE DATA: Creating telemetry.csv")
    print("="*60)
    print("📅 Generating 90 days of data.")
    df = generate_telemetry_data(days=90, samples_per_day=1440)
    df.to_csv(output_path, index=False)
    print(f"✅ Created {len(df):,} samples at {output_path}")
    print(f"📅 Path: {output_path}")
    return output_path

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SATELLITE TELEMETRY DATA GENERATOR")
    print("="*60 + "\n")
    
    # Generate 90 days of data with 1 sample per minute
    df = generate_telemetry_data(days=90, samples_per_day=1440)
    
    # Save to CSV
    output_path = '/opt/airflow/dags/data/telemetry.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nGenerated {len(df):,} telemetry samples")
    print(f"Saved to {output_path}")
    print(f"Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].sum()/len(df)*100:.2f}%)")
    
    # Show first few rows
    print("\n" + "="*60)
    print("First 5 rows:")
    print("="*60)
    print(df.head())
    
    # Show data summary
    print("\n" + "="*60)
    print("Data Summary:")
    print("="*60)
    print(df.describe())
    
    # Show anomaly breakdown
    print("\n" + "="*60)
    print("Anomaly Statistics:")
    print("="*60)
    print(f"Normal samples: {(df['is_anomaly'] == 0).sum():,}")
    print(f"Anomalous samples: {(df['is_anomaly'] == 1).sum():,}")
    
    print("\nData generation complete!")
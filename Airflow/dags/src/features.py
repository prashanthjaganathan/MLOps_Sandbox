"""
Feature engineering for satellite telemetry anomaly detection

This module creates derived features that help ML models detect anomalies:
- Rolling statistics (trends over time)
- Rate of change (sudden spikes)
- Derived metrics (power = voltage × current)
"""

import pandas as pd
import numpy as np

def engineer_features(input_path, output_path):
    """
    Create features for anomaly detection
    
    Features we create:
    1. Rolling mean (24-hour average) - captures trends
    2. Rolling std (24-hour variability) - captures instability
    3. Rate of change (difference from previous hour) - captures sudden changes
    4. Power calculation (voltage × current) - derived metric
    
    Args:
        input_path: Path to cleaned CSV
        output_path: Path to save features CSV
    
    Returns:
        str: Path to output file
    """
    print("="*60)
    print("⚙️  FEATURE ENGINEERING: Creating ML features")
    print("="*60)
    
    # Load cleaned data
    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df):,} samples")
    
    initial_columns = list(df.columns)
    print(f"📊 Initial columns: {initial_columns}")
    
    # ============================================
    # FEATURE 1: Rolling Statistics (24-hour window)
    # ============================================
    print("\n🔄 Creating rolling statistics (24-hour window)...")
    
    # Temperature rolling mean and std
    df['temp_rolling_mean'] = df['temperature'].rolling(window=24, min_periods=1).mean()
    df['temp_rolling_std'] = df['temperature'].rolling(window=24, min_periods=1).std()
    
    # Voltage rolling mean
    df['voltage_rolling_mean'] = df['voltage'].rolling(window=24, min_periods=1).mean()
    
    # Current rolling mean
    df['current_rolling_mean'] = df['current'].rolling(window=24, min_periods=1).mean()
    
    print(f"✅ Created rolling mean features")
    print(f"✅ Created rolling std features")
    
    # ============================================
    # FEATURE 2: Rate of Change (Deltas)
    # ============================================
    print("\n📈 Creating rate of change features...")
    
    # Temperature change from previous hour
    df['temp_change'] = df['temperature'].diff()
    
    # Voltage change from previous hour
    df['voltage_change'] = df['voltage'].diff()
    
    print(f"✅ Created delta features")
    
    # ============================================
    # FEATURE 3: Derived Metrics
    # ============================================
    print("\n⚡ Creating derived features...")
    
    # Power = Voltage × Current (Watts)
    df['power'] = df['voltage'] * df['current']
    
    print(f"✅ Created power feature")
    
    # ============================================
    # Handle NaN values from rolling/diff operations
    # ============================================
    null_count_before = df.isnull().sum().sum()
    if null_count_before > 0:
        print(f"\n⚠️  Found {null_count_before} NaN values from feature creation")
        print(f"🔧 Applying backward fill...")
        df = df.fillna(method='bfill')
        print(f"✅ Filled NaN values")
    
    # ============================================
    # Save features
    # ============================================
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved features to: {output_path}")
    
    # Summary
    new_columns = [col for col in df.columns if col not in initial_columns]
    print(f"\n📊 Feature Engineering Summary:")
    print(f"  Original features: {len(initial_columns)}")
    print(f"  New features: {len(new_columns)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"\n  New features created:")
    for col in new_columns:
        print(f"    - {col}")
    
    print("\n✅ Feature engineering complete!")
    return output_path
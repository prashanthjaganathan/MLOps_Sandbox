"""
Data preprocessing functions for satellite telemetry

This module handles:
- Data cleaning (removing duplicates, handling nulls)
- Resampling (converting per-minute data to hourly)
- Basic validation
"""

import pandas as pd
import numpy as np

def clean_and_resample(input_path, output_path):
    """
    Clean data and resample to hourly intervals
    
    Why resample?
    - Reduces data size (1440 samples/day → 24 samples/day)
    - Reduces noise (averaging smooths out random fluctuations)
    - Faster model training
    
    Args:
        input_path: Path to raw telemetry CSV
        output_path: Path to save cleaned CSV
    
    Returns:
        str: Path to output file
    """
    print("="*60)
    print("🧹 PREPROCESSING: Cleaning and resampling data")
    print("="*60)
    
    # Load raw data
    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df):,} raw samples")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Converted timestamps to datetime")
    
    # Remove duplicates (if any)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['timestamp'])
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        print(f"🗑️  Removed {duplicates_removed} duplicate timestamps")
    else:
        print(f"✅ No duplicates found")
    
    # Set timestamp as index (required for resampling)
    df = df.set_index('timestamp')
    
    # Resample to hourly intervals (mean of each hour)
    print(f"⏱️  Resampling from per-minute to hourly...")
    df_resampled = df.resample('1H').mean()
    
    # Make is_anomaly binary again: 1 if any minute in that hour was anomalous
    if 'is_anomaly' in df.columns:
        df_resampled['is_anomaly'] = (df_resampled['is_anomaly'] > 0).astype(int)
    
    print(f"✅ Resampled: {len(df):,} samples → {len(df_resampled):,} samples")
    print(f"📉 Data reduction: {(1 - len(df_resampled)/len(df))*100:.1f}%")
    
    # Handle missing values (forward fill)
    # This fills gaps by carrying forward the last known value
    null_before = df_resampled.isnull().sum().sum()
    if null_before > 0:
        print(f"⚠️  Found {null_before} null values, applying forward fill...")
        df_resampled = df_resampled.fillna(method='ffill')
        print(f"✅ Filled null values")
    else:
        print(f"✅ No null values to handle")
    
    # Reset index to make timestamp a column again
    df_resampled = df_resampled.reset_index()
    
    # Save cleaned data
    df_resampled.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned data to: {output_path}")
    
    # Summary statistics
    print("\n📊 Cleaned Data Summary:")
    print(f"  Samples: {len(df_resampled):,}")
    print(f"  Columns: {list(df_resampled.columns)}")
    print(f"  Date range: {df_resampled['timestamp'].min()} to {df_resampled['timestamp'].max()}")
    
    print("\n✅ Preprocessing complete!")
    return output_path
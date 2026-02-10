"""
Model scoring for satellite telemetry anomaly detection

This module uses a trained model to predict anomalies in telemetry data
"""

import pandas as pd
import pickle
import numpy as np

def score_telemetry(input_path, model_path, output_path):
    """
    Score telemetry data for anomalies using trained model
    
    The model returns:
    - Prediction: -1 (anomaly) or 1 (normal)
    - Score: Anomaly score (lower = more anomalous)
    
    Args:
        input_path: Path to features CSV
        model_path: Path to trained model (.pkl)
        output_path: Path to save scored CSV
    
    Returns:
        str: Path to output file
    """
    print("="*60)
    print("🎯 MODEL SCORING: Detecting anomalies")
    print("="*60)
    
    # Load features
    print(f"📂 Loading features from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df):,} samples")
    
    # Load trained model
    print(f"📂 Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully")
    
    # ============================================
    # Select features (same as training)
    # ============================================
    feature_cols = [
        'temperature',
        'voltage',
        'current',
        'angular_velocity',
        'solar_power',
        'temp_rolling_mean',
        'voltage_rolling_mean',
        'current_rolling_mean',
        'power',
    ]
    
    X = df[feature_cols]
    print(f"✅ Feature matrix shape: {X.shape}")
    
    # ============================================
    # Predict anomalies
    # ============================================
    print(f"\n🔮 Predicting anomalies...")
    
    # Predict: -1 = anomaly, 1 = normal
    predictions = model.predict(X)
    
    # Get anomaly scores (lower = more anomalous)
    scores = model.score_samples(X)
    
    print(f"✅ Predictions complete!")
    
    # ============================================
    # Add predictions to dataframe
    # ============================================
    df['anomaly_prediction'] = predictions
    df['anomaly_score'] = scores
    df['is_predicted_anomaly'] = (predictions == -1).astype(int)
    
    # ============================================
    # Analyze results
    # ============================================
    anomaly_count = df['is_predicted_anomaly'].sum()
    normal_count = len(df) - anomaly_count
    
    print(f"\n📊 Scoring Results:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Predicted anomalies: {anomaly_count:,} ({anomaly_count/len(df)*100:.2f}%)")
    print(f"  Predicted normal: {normal_count:,} ({normal_count/len(df)*100:.2f}%)")
    
    # Show most anomalous samples
    print(f"\n🚨 Top 5 Most Anomalous Samples:")
    top_anomalies = df.nsmallest(5, 'anomaly_score')[['timestamp', 'temperature', 'voltage', 'anomaly_score']]
    print(top_anomalies.to_string(index=False))
    
    # ============================================
    # Save scored data
    # ============================================
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved scored data to: {output_path}")
    
    print("\n✅ Scoring complete!")
    return output_path
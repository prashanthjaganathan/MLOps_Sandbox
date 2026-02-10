"""
Model training for satellite telemetry anomaly detection

This module trains an Isolation Forest model to detect anomalies
"""

import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest

def train_anomaly_model(input_path, model_path):
    """
    Train Isolation Forest for anomaly detection
    
    Isolation Forest parameters:
    - contamination: Expected proportion of anomalies (0.05 = 5%)
    - n_estimators: Number of trees (100 is a good default)
    - random_state: For reproducibility
    
    Args:
        input_path: Path to features CSV
        model_path: Path to save trained model (.pkl)
    
    Returns:
        str: Path to saved model
    """
    print("="*60)
    print("🤖 MODEL TRAINING: Training Isolation Forest")
    print("="*60)
    
    # Load features
    print(f"📂 Loading features from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} features")
    
    # ============================================
    # Select features for training
    # ============================================
    # We use both original and engineered features
    feature_cols = [
        # Original features
        'temperature',
        'voltage',
        'current',
        'angular_velocity',
        'solar_power',
        # Engineered features
        'temp_rolling_mean',
        'voltage_rolling_mean',
        'current_rolling_mean',
        'power',
    ]
    
    print(f"\n📊 Training features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    # Extract feature matrix
    X = df[feature_cols]
    print(f"\n✅ Feature matrix shape: {X.shape}")
    
    # ============================================
    # Train Isolation Forest
    # ============================================
    print(f"\n🌲 Training Isolation Forest...")
    print(f"  Parameters:")
    print(f"    - contamination: 0.05 (expect 5% anomalies)")
    print(f"    - n_estimators: 100 (number of trees)")
    print(f"    - random_state: 42 (for reproducibility)")
    
    model = IsolationForest(
        contamination=0.05,  # Expect 5% anomalies
        random_state=42,     # For reproducibility
        n_estimators=100,    # Number of trees in forest
        n_jobs=-1            # Use all CPU cores
    )
    
    # Fit the model
    model.fit(X)
    print(f"✅ Model training complete!")
    
    # ============================================
    # Model evaluation (on training data)
    # ============================================
    print(f"\n📊 Model Evaluation:")
    
    # Predict on training data
    predictions = model.predict(X)
    anomaly_count = (predictions == -1).sum()
    
    print(f"  Samples: {len(X):,}")
    print(f"  Predicted anomalies: {anomaly_count:,} ({anomaly_count/len(X)*100:.2f}%)")
    print(f"  Predicted normal: {(predictions == 1).sum():,} ({(predictions == 1).sum()/len(X)*100:.2f}%)")
    
    # ============================================
    # Save model
    # ============================================
    print(f"\n💾 Saving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved successfully!")
    print(f"📦 Model size: {len(pickle.dumps(model)) / 1024:.2f} KB")
    
    print("\n✅ Training complete!")
    return model_path
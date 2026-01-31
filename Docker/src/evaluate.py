import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model_path='models/sentiment_model.pkl', test_data_path='data/test_data.csv'):
    """Evaluate the trained model"""
    
    print("Loading model...")
    model = joblib.load(model_path)
    
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    
    X_test = test_df['review']
    y_test = test_df['sentiment']
    
    # Predict
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

if __name__ == "__main__":
    evaluate_model()
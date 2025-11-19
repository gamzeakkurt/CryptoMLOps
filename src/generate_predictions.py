"""
Script to generate predictions using the trained model.
Loads data, creates features, and generates predictions for all available data points.
"""
import pandas as pd
import numpy as np
from joblib import load
from data_pipeline import load_data, make_features
import os
from datetime import datetime
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

def generate_predictions(data_path=None, 
                        model_path=None,
                        output_path=None,
                        window=24):
    """
    Generate predictions for all data points in the dataset.
    
    Args:
        data_path: Path to the CSV file with crypto data
        model_path: Path to the trained model file
        output_path: Path where predictions will be saved
        window: Number of lag features (should match training)
    
    Returns:
        DataFrame with predictions and original data
    """
    # Set default paths relative to script directory
    if data_path is None:
        data_path = SCRIPT_DIR / 'data' / 'BTCUSDT_6months_hourly.csv'
    if model_path is None:
        model_path = SCRIPT_DIR / 'model.pkl'
    if output_path is None:
        output_path = SCRIPT_DIR / 'predictions.csv'
    
    # Convert to strings for compatibility
    data_path = str(data_path)
    model_path = str(model_path)
    output_path = str(output_path)
    
    print("🔄 Loading data...")
    # Use absolute path for data loading
    df = load_data(data_path)
    print(f"✅ Loaded {len(df)} rows of data")
    
    print("🔄 Creating features...")
    X, y = make_features(df, window=window)
    print(f"✅ Created features for {len(X)} samples")
    
    print("🔄 Loading model...")
    try:
        model = load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    print("🔄 Generating predictions...")
    predictions = model.predict(X)
    print(f"✅ Generated {len(predictions)} predictions")
    
    # Create results DataFrame
    # Get the corresponding timestamps (after dropping NaN rows from feature creation)
    result_df = df.iloc[window:].copy()  # Skip first 'window' rows that were dropped
    result_df = result_df.reset_index(drop=True)
    
    # Add predictions and actual values
    result_df['predicted_return'] = predictions
    result_df['actual_return'] = y.values
    
    # Calculate prediction error
    result_df['prediction_error'] = result_df['actual_return'] - result_df['predicted_return']
    result_df['absolute_error'] = np.abs(result_df['prediction_error'])
    
    # Calculate predicted price (if needed)
    # predicted_return is log return, so: predicted_price = close * exp(predicted_return)
    result_df['predicted_price'] = result_df['close'] * np.exp(result_df['predicted_return'])
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    # Print summary statistics
    print("\n📊 Prediction Summary:")
    print(f"   Mean Absolute Error: {result_df['absolute_error'].mean():.6f}")
    print(f"   Root Mean Squared Error: {np.sqrt((result_df['prediction_error']**2).mean()):.6f}")
    print(f"   Mean Actual Return: {result_df['actual_return'].mean():.6f}")
    print(f"   Mean Predicted Return: {result_df['predicted_return'].mean():.6f}")
    print(f"   Correlation: {result_df['actual_return'].corr(result_df['predicted_return']):.4f}")
    
    return result_df


def generate_sample_predictions(n_samples=10, data_path=None, 
                                model_path=None, window=24):
    """
    Generate sample predictions with feature values that can be used to test the API.
    
    Args:
        n_samples: Number of sample predictions to generate
        data_path: Path to the data CSV file
        model_path: Path to the trained model file
        window: Number of lag features
    
    Returns:
        List of dictionaries with features and predictions
    """
    # Set default paths relative to script directory
    if data_path is None:
        data_path = SCRIPT_DIR / 'data' / 'BTCUSDT_6months_hourly.csv'
    if model_path is None:
        model_path = SCRIPT_DIR / 'model.pkl'
    
    # Convert to strings for compatibility
    data_path = str(data_path)
    model_path = str(model_path)
    
    print("🔄 Loading data for sample generation...")
    df = load_data(data_path)
    X, y = make_features(df, window=window)
    
    print("🔄 Loading model...")
    try:
        model = load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Take last n_samples from the data
    sample_X = X.tail(n_samples)
    sample_y = y.tail(n_samples)
    
    print("🔄 Generating sample predictions...")
    predictions = model.predict(sample_X)
    
    # Create list of feature dictionaries
    samples = []
    for idx, (features, actual, predicted) in enumerate(zip(sample_X.iterrows(), sample_y, predictions)):
        row_idx, feature_row = features
        feature_dict = feature_row.to_dict()
        samples.append({
            'sample_id': idx + 1,
            'features': feature_dict,
            'predicted_return': float(predicted),
            'actual_return': float(actual),
            'prediction_error': float(actual - predicted)
        })
    
    return samples


def save_predictions_json(predictions_df, output_path='predictions.json'):
    """
    Save predictions in JSON format for API testing.
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save JSON file
    """
    # Select relevant columns and convert to JSON
    json_data = predictions_df[['timestamp', 'close', 'actual_return', 'predicted_return', 
                                'prediction_error', 'predicted_price']].to_dict('records')
    
    import json
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"✅ Predictions saved to {output_path} in JSON format")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions from crypto data')
    parser.add_argument('--data', type=str, default='data/BTCUSDT_6months_hourly.csv',
                       help='Path to input data CSV')
    parser.add_argument('--model', type=str, default='model.pkl',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to save predictions CSV')
    parser.add_argument('--json-output', type=str, default='predictions.json',
                       help='Path to save predictions JSON')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of sample predictions to generate for API testing')
    parser.add_argument('--window', type=int, default=24,
                       help='Number of lag features')
    
    args = parser.parse_args()
    
    # Generate full predictions
    print("=" * 60)
    print("Generating Predictions")
    print("=" * 60)
    predictions_df = generate_predictions(
        data_path=args.data,
        model_path=args.model,
        output_path=args.output,
        window=args.window
    )
    
    if predictions_df is not None:
        # Save JSON version
        save_predictions_json(predictions_df, args.json_output)
        
        # Generate sample predictions for API testing
        print("\n" + "=" * 60)
        print("Generating Sample Predictions for API Testing")
        print("=" * 60)
        samples = generate_sample_predictions(
            n_samples=args.samples,
            data_path=args.data,
            model_path=args.model,
            window=args.window
        )
        
        if samples:
            # Save samples to JSON
            import json
            samples_output = 'sample_predictions.json'
            with open(samples_output, 'w') as f:
                json.dump(samples, f, indent=2)
            print(f"✅ Sample predictions saved to {samples_output}")
            
            # Print first sample as example
            print("\n📝 Example API Request (first sample):")
            print(json.dumps(samples[0]['features'], indent=2))
    
    print("\n✅ All predictions generated successfully!")


import pandas as pd
import xgboost as xgb
import joblib
import argparse


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data, scaler):
    X = data
    X_scaled = scaler.transform(X)
    return X_scaled


def load_model(model_path):
    model = joblib.load(model_path)
    return model


def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler


def save_predictions(predictions, output_path):
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a trained XGBoost model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the test data file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model.")
    parser.add_argument('--scaler_path', type=str, required=True, help="Path to the scaler.")
    parser.add_argument('--output_path', type=str, default="predictions.csv", help="Path to save the predictions.")

    args = parser.parse_args()

    data = load_data(args.data_path)
    scaler = load_scaler(args.scaler_path)
    X_scaled = preprocess_data(data, scaler)
    model = load_model(args.model_path)

    predictions = model.predict(X_scaled)
    save_predictions(predictions, args.output_path)
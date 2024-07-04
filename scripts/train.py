import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_model(X, y):
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                               n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation RMSE: {np.sqrt(-grid_search.best_score_)}")

    best_model = grid_search.best_estimator_
    return best_model


def save_model(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XGBoost model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the training data file.")
    parser.add_argument('--model_path', type=str, default="xgboost_model.pkl", help="Path to save the trained model.")
    parser.add_argument('--scaler_path', type=str, default="scaler.pkl", help="Path to save the scaler.")

    args = parser.parse_args()

    data = load_data(args.data_path)
    X, y, scaler = preprocess_data(data)
    model = train_model(X, y)
    save_model(model, scaler, args.model_path, args.scaler_path)
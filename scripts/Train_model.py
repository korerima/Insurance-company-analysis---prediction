import pandas as pd
import yaml
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import argparse

def predict_premium(input_path, param_path, metrics_path):
    # Load data
    df = pd.read_csv(input_path)

    # Feature engineering
    df['TotalPremium'] = df['CalculatedPremiumPerTerm'] * 0.8772
    df['ClaimToPremiumRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-6)  # Avoid divide-by-zero
    df['VehicleAge'] = 2015 - df['RegistrationYear']

    # Load model parameters
    with open(param_path, 'r') as param_file:
        params = yaml.safe_load(param_file)
        prem_param = params['Train_model']['premium']

    # Select features
    features = [
        'make', 'cubiccapacity', 'kilowatts', 'NewVehicle', 'ExcessSelected',
        'CoverType', 'Product', 'LegalType', 'Province', 'IsVATRegistered',
        'TotalClaims', 'TotalPremium'
    ]

    # Prepare data
    df_new = pd.DataFrame()
    df_new[features] = df[features]
    categorical_features = ['make', 'NewVehicle', 'ExcessSelected', 'CoverType', 'Product', 'LegalType', 'Province']
    df_new = pd.get_dummies(df_new, columns=categorical_features, drop_first=True)

    X = df_new.drop('TotalPremium', axis=1)
    y = df['TotalPremium']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost Regressor
    xgb_model = xgb.XGBRegressor(**prem_param)
    xgb_model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = xgb_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics
    metrics = {
        'mean_squared_error': mse,
        'r2_score': r2
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate premium prediction model")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--params', type=str, required=True, help="Path to parameter YAML file")
    parser.add_argument('--metrics', type=str, required=True, help="Path to output metrics JSON file")
    args = parser.parse_args()

    predict_premium(args.input, args.params, args.metrics)

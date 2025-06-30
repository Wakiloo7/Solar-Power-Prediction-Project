import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

MAX_POWER = 1.6 * 1000 * 0.96  # 1536 watts

def custom_mape(y_true, y_pred):
    mask = y_true > 1.0
    if not np.any(mask):
        return 0.0
    offset = 0.01
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    denominator = np.maximum(y_true_masked + offset, 1.0)
    return np.mean(np.abs((y_true_masked - y_pred_masked) / denominator)) * 100

def train_xgboost_model(df):
    print("Training XGBoost Model...")
    print(f"XGBoost Version: {xgb.__version__}")

    features = ['temp', 'swflx', 'hour', 'day_of_year', 'month', 'is_daytime',
                'temp_swflx_interaction', 'hour_sin', 'hour_cos', 'swflx_lag1',
                'swflx_lag24', 'temp_lag1', 'swflx_rolling_mean', 'swflx_rolling_mean_6',
                'temp_rolling_mean', 'is_weekend', 'prod_inj_lag24', 'solar_zenith_angle',
                'prod_inj_lag1', 'swflx_rolling_max_6', 'swflx_peak_indicator']

    X = df[features].values
    y = df['power_watts'].values

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample high power in training
    high_power_mask = y_train > 20
    X_high = X_train[high_power_mask]
    y_high = y_train[high_power_mask]
    X_train = np.vstack([X_train] + [X_high] * 5)
    y_train = np.hstack([y_train] + [y_high] * 5)

    # Scaling
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Model
    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        early_stopping_rounds=100,
        objective='reg:squarederror',
        random_state=42,
        verbosity=1
    )

    model.fit(
        X_train_scaled, y_train_scaled,
        eval_set=[(X_test_scaled, y_test_scaled)],
        eval_metric='mae',
        verbose=100
    )

    # Evaluate on train
    y_pred_train_scaled = model.predict(X_train_scaled)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred_train = np.clip(y_pred_train, 0, None)
    train_mape = custom_mape(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    print(f"\nXGBoost Training Fit:")
    print(f"Train MAPE: {train_mape:.2f}%")
    print(f"Train MAE: {train_mae:.4f} watts")

    # Evaluate on test
    y_pred_test_scaled = model.predict(X_test_scaled)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
    y_pred_test = np.clip(y_pred_test, 0, None)
    test_mape = custom_mape(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\nXGBoost Test Fit:")
    print(f"Test MAPE: {test_mape:.2f}%")
    print(f"Test MAE: {test_mae:.4f} watts")

    # Extract timestamps
    test_indices = df.index[train_test_split(df.index, test_size=0.2, random_state=42)[1]]
    timestamps_test = df.loc[test_indices, 'timestamp'].reset_index(drop=True)

    return model, scaler_X, scaler_y, X_test, y_test, timestamps_test

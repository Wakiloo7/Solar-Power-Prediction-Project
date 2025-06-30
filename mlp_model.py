import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance

# PV system parameters
MAX_POWER = 1.6 * 1000 * 0.96  # 1536 watts

def custom_mape(y_true, y_pred):
    """Custom MAPE for non-zero actual values with higher threshold."""
    mask = (y_true > 1.0)
    if not np.any(mask):
        print("Warning: No actual values > 1.0 for MAPE calculation, returning 0")
        return 0.0
    y_true_day = y_true[mask]
    y_pred_day = y_pred[mask]
    offset = 0.01
    denominator = np.maximum(y_true_day + offset, 1.0)
    mape = np.mean(np.abs((y_true_day - y_pred_day) / denominator)) * 100
    return mape

def train_mlp_model(df):
    print("Training MLP Model...")

    features = ['temp', 'swflx', 'hour', 'day_of_year', 'month', 'is_daytime', 
                'temp_swflx_interaction', 'hour_sin', 'hour_cos', 'swflx_lag1', 
                'swflx_lag24', 'temp_lag1', 'swflx_rolling_mean', 'swflx_rolling_mean_6', 
                'temp_rolling_mean', 'is_weekend', 'prod_inj_lag24', 'solar_zenith_angle', 
                'prod_inj_lag1', 'swflx_rolling_max_6', 'swflx_peak_indicator']
    
    X = df[features].values
    y = df['power_watts'].values

    # Split once: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample high power in training only
    high_power_mask = y_train > 20
    X_high = X_train[high_power_mask]
    y_high = y_train[high_power_mask]
    X_train = np.vstack([X_train] + [X_high]*5)
    y_train = np.hstack([y_train] + [y_high]*5)

    # Scaling
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(300, 150, 75),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=3000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=100,
        alpha=0.001,
        tol=1e-5
    )

    model.fit(X_train_scaled, y_train_scaled)

    def evaluate(split_name, X_scaled, y_true, y_scaled_true):
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        mape = custom_mape(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"\n{split_name} MAPE: {mape:.2f}%")
        print(f"{split_name} MAE: {mae:.2f} watts")
        return y_pred

    print("\nPerformance Evaluation:")
    y_pred_train = evaluate("Train", X_train_scaled, y_train, y_train_scaled)
    y_pred_test = evaluate("Test", X_test_scaled, y_test, y_test_scaled)

    print("\nSample Predictions:")
    print(pd.DataFrame({
        'Actual': y_test[:10],
        'Predicted': np.clip(y_pred_test[:10], 0, None)
    }))

    # Timestamp alignment
    test_indices = df.index[train_test_split(df.index, test_size=0.2, random_state=42)[1]]
    timestamps_test = df.loc[test_indices, 'timestamp'].reset_index(drop=True)

    return model, scaler_X, scaler_y, X_test, y_test, timestamps_test



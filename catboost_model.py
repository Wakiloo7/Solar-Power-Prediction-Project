import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# PV system parameters
LATITUDE = 41.12
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

def train_catboost_model(df):
    """Train CatBoost to predict power_watts for 15-minute interval data."""
    import catboost
    print(f"CatBoost Version: {catboost.__version__}")
    
    df = df.copy()
    
    features = ['temp', 'swflx', 'hour', 'day_of_year', 'month', 'is_daytime', 
                'temp_swflx_interaction', 'hour_sin', 'hour_cos', 'swflx_lag1', 
                'swflx_lag24', 'temp_lag1', 'swflx_rolling_mean', 'swflx_rolling_mean_6', 
                'temp_rolling_mean', 'is_weekend', 'prod_inj_lag24', 'solar_zenith_angle']
    
    X = df[features].values
    y = df['power_watts'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=5000,
        depth=8,
        learning_rate=0.02,
        loss_function='MAE',
        random_seed=42,
        verbose=100,
        subsample=0.9,
        colsample_bylevel=0.8,
        min_child_samples=3,
        l2_leaf_reg=5,
        bagging_temperature=0.5,
        early_stopping_rounds=200,
        border_count=128,
        random_strength=1.0,
        grow_policy='Lossguide',
        max_leaves=64
    )
    
    print("CatBoost Model Parameters:")
    print(model.get_params())
    
    model.fit(X_train, y_train, verbose=100)

    # Evaluate training performance
    y_pred_train = model.predict(X_train)
    y_pred_train = np.where(X_train[:, 5] == 0, 0, y_pred_train)  # is_daytime == 0 -> zero
    y_pred_train = np.clip(y_pred_train, 0, None)

    train_mape = custom_mape(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    print("\nCatBoost Feature Importance:")
    print(pd.DataFrame({
        'feature': features,
        'importance': model.get_feature_importance()
    }).sort_values(by='importance', ascending=False))

    print("\nCatBoost Training Fit Check (first 5 daytime values):")
    daytime_idx = np.where(y_train > 0)[0][:5]
    print(f"Actual: {y_train[daytime_idx]}")
    print(f"Predicted: {y_pred_train[daytime_idx]}")
    print(f"MAPE on training data (15-min intervals): {train_mape:.2f}%")
    print(f"MAE on training data: {train_mae:.4f} watts")

    # Reconstruct test timestamps
    test_indices = df.index[train_test_split(df.index, test_size=0.2, random_state=42)[1]]
    timestamps = df.loc[test_indices, 'timestamp'].reset_index(drop=True)

    # Return scalers as None to match main.py expectations
    return model, None, None, X_test, y_test, timestamps

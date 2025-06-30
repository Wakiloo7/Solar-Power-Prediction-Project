import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import joblib

from xgboost_model import train_xgboost_model
from input import preprocess_data, MAX_POWER
from catboost_model import train_catboost_model, custom_mape
from mlp_model import train_mlp_model, custom_mape

def hourly_mape(y_true, y_pred):
    mape_values = []
    for true, pred in zip(y_true, y_pred):
        if true > 1.0:
            mape = np.abs((true - pred) / (true + 0.01)) * 100
        else:
            mape = 0.0
        mape_values.append(mape)
    return np.array(mape_values)

def predict_test_set(model, X_test, y_test, timestamps, scaler_X, scaler_y, model_type='mlp', example_date='2025-01-20'):
    X_pred_scaled = X_test if scaler_X is None else scaler_X.transform(X_test)
    predictions_scaled = model.predict(X_pred_scaled)

    # Invert scaling for y
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
    else:
        predictions = predictions_scaled

    predictions = np.clip(predictions, 0, None)  # Remove negative predictions

    # MAPE per sample
    mape_values = hourly_mape(y_test, predictions)

    results = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_power_watts': predictions,
        'actual_power_watts': y_test,
        'mape': mape_values
    })

    results.to_csv(f'test_predictions_{model_type.lower()}.csv', index=False)
    print(f"\n{model_type.upper()} Test Set Predictions (first 10 rows):")
    print(results[['timestamp', 'predicted_power_watts', 'actual_power_watts', 'mape']].head(10))

    overall_mape = custom_mape(y_test, predictions)
    overall_mae = mean_absolute_error(y_test, predictions)
    print(f"\n{model_type.upper()} Overall Test Set Performance:")
    print(f"Average MAPE: {overall_mape:.2f}%")
    print(f"Average MAE: {overall_mae:.4f} watts")

    # Example day
    example_date = pd.to_datetime(example_date).date()
    day_results = results[results['timestamp'].dt.date == example_date]

    if not day_results.empty:
        print(f"\n{model_type.upper()} Predictions for {example_date}:")
        print(day_results[['timestamp', 'predicted_power_watts', 'actual_power_watts', 'mape']])
        print(f"Total actual power: {day_results['actual_power_watts'].sum():.2f} watts")
        print(f"Total predicted power: {day_results['predicted_power_watts'].sum():.2f} watts")
        print(f"Average MAPE: {day_results['mape'].mean():.2f}%")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(day_results['timestamp'], day_results['actual_power_watts'], label='Actual Power', marker='o', markersize=4)
        plt.plot(day_results['timestamp'], day_results['predicted_power_watts'], label=f'Predicted ({model_type.upper()})', marker='x', markersize=4)
        plt.title(f'Power for {example_date} - 15min ({model_type.upper()})')
        plt.xlabel('Timestamp')
        plt.ylabel('Power (watts)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{example_date}_power_plot_{model_type.lower()}_15min.png')
        plt.close()
    else:
        print(f"Warning: No test data for {example_date}")

    return results


def main():
    csv_file = r"C:\Users\md.w.ahmad\PythonProjects\Sentinal_weather-data\sentinel_client\15min_interval_power_calc\processed_sgl_v2_data.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return

    df = preprocess_data(df)
    
    print("\nTraining XGBoost Model...")
    xgb_model, xgb_scaler_X, xgb_scaler_y, X_test_xgb, y_test_xgb, timestamps_xgb = train_xgboost_model(df)
    joblib.dump(xgb_model, 'solar_power_model_xgboost_15min.pkl')
    predict_test_set(xgb_model, X_test_xgb, y_test_xgb, timestamps_xgb, xgb_scaler_X, xgb_scaler_y, model_type='xgboost')


    print("\nTraining MLP Model...")
    mlp_model, mlp_scaler_X, mlp_scaler_y, X_test_mlp, y_test_mlp, timestamps_mlp = train_mlp_model(df)
    joblib.dump(mlp_model, 'solar_power_model_mlp_15min.pkl')
    predict_test_set(mlp_model, X_test_mlp, y_test_mlp, timestamps_mlp, mlp_scaler_X, mlp_scaler_y, model_type='mlp')


    print("\nTraining CatBoost Model...")
    catboost_model, catboost_scaler_X, catboost_scaler_y, X_test_cat, y_test_cat, timestamps_cat = train_catboost_model(df)
    catboost_model.save_model('solar_power_model_catboost_15min.cb')
    predict_test_set(catboost_model, X_test_cat, y_test_cat, timestamps_cat, catboost_scaler_X, catboost_scaler_y, model_type='catboost')

if __name__ == "__main__":
    main()

Solar Power Prediction Project
This project predicts solar power output using three machine learning models: XGBoost, MLP (Neural Network), and CatBoost. The models use weather and time data to predict power output in watts for a solar panel system with a maximum capacity of 1536 watts. The data is based on 15-minute intervals from April 3, 2024, to January 21, 2025.
Results Summary
We measure the performance of the models using Mean Absolute Percentage Error (MAPE). MAPE shows how accurate the predictions are compared to actual values, focusing on daytime data (when power is greater than 1 watt).
Model Performance on Test Data

XGBoost:
Test MAPE: 15.95%
For January 20, 2025, average MAPE: 15.11%
Total actual power on January 20, 2025: 27.26 watts
Total predicted power: 34.19 watts


MLP (Neural Network):
Test MAPE: 16.55%
For January 20, 2025, average MAPE: 11.48%
Total actual power on January 20, 2025: 27.26 watts
Total predicted power: 30.78 watts


CatBoost:
Test MAPE: 20.26%
For January 20, 2025, average MAPE: 19.80%
Total actual power on January 20, 2025: 27.26 watts
Total predicted power: 32.13 watts



Key Observations

XGBoost performs best overall with the lowest test MAPE (15.95%).
MLP has the best MAPE for January 20, 2025 (11.48%), showing good performance for that specific day.
CatBoost has the highest test MAPE (20.26%), indicating it is less accurate compared to XGBoost and MLP.
All models predict higher total power than actual values for January 20, 2025.

How to Use

Install required libraries: numpy, pandas, scikit-learn, xgboost, catboost, matplotlib.
Place the input CSV file in the project folder.
Run main.py to train models and generate predictions.
Results are saved as CSV files and plots for each model.

Notes

The models use features like temperature, solar flux, hour, and solar zenith angle.
Data is preprocessed to handle missing values and resampled to 15-minute intervals.
Predictions are clipped to avoid negative values.

For more details, check the code and output files in the repository.

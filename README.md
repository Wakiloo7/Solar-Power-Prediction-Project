# Solar Power Prediction

This project uses machine learning to predict solar power output for a 1536-watt solar panel system. It employs three models—XGBoost, MLP (Neural Network), and CatBoost—to forecast power in watts based on 15-minute interval data from April 3, 2024, to January 21, 2025. The models use weather and time-based features like temperature, solar flux, and solar zenith angle.

## Results

Performance is evaluated using **Mean Absolute Percentage Error (MAPE)**, focusing on daytime data (power > 1 watt).

### Test Set Performance
- **XGBoost**:
  - Test MAPE: **15.95%** for January 20, 2025, average MAPE: 15.11%
  - Total actual power on January 20, 2025: **27.26 watts**
  - Total predicted power: **34.19 watts**
- **MLP (Neural Network)**:
  - Test MAPE: **16.55%** for January 20, 2025, average MAPE: 11.48%
  - Total actual power on January 20, 2025: **27.26 watts**
  - Total predicted power: **30.78 watts**
- **CatBoost**:
  - Test MAPE: **20.26%** for January 20, 2025, average MAPE: 19.80%
  - Total actual power on January 20, 2025: **27.26 watts**
  - Total predicted power: **32.13 watts**

### Key Observations
- **XGBoost** performs best overall with the lowest test MAPE (15.95%).
- **MLP** achieves the best MAPE for January 20, 2025 (11.48%), showing good performance for that specific day.
- **CatBoost** has the highest test MAPE (20.26%), indicating lower accuracy.
- All models predict higher total power than actual values for January 20, 2025.

## Installation

1. Clone the repository:
   ```bash
    https://github.com/Wakiloo7/Solar-Power-Prediction-Project.git

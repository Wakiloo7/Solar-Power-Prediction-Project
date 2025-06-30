import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# PV system parameters
LATITUDE = 41.12
LONGITUDE = -8.65
ALTITUDE = 23  # meters
CAPACITY_KW = 1.6  # kW
ARRAY_TILT = 35  # degrees
ARRAY_AZIMUTH = 180  # degrees (south)
INVERTER_EFFICIENCY = 0.96
MAX_POWER = CAPACITY_KW * 1000 * INVERTER_EFFICIENCY  # 1536 watts

def preprocess_data(df):
    """Process CSV data with deduplication, resample to 15-minute intervals, and use Production_Injection_C424 as power in watts."""
    df = df.copy()
    
    # Ensure required columns exist
    required_columns = ['datetime', 'temp', 'swflx', 'Production_Injection_C424']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Ensure datetime is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter data to April 3, 2024, to January 21, 2025
    start_date = pd.to_datetime('2024-04-03')
    end_date = pd.to_datetime('2025-01-21')
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    
    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
    
    # Resample to 15-minute intervals, aggregating with mean for numeric columns
    df = df.set_index('datetime').resample('15min').mean(numeric_only=True).reset_index()
    
    # Forward-fill and back-fill missing values
    df = df.ffill().bfill()
    
    # Rename datetime to timestamp
    df = df.rename(columns={'datetime': 'timestamp'})
    
    # Clip unrealistic values
    df['temp'] = df['temp'].clip(lower=-50, upper=50)
    df['swflx'] = df['swflx'].clip(lower=0, upper=1500)
    df['Production_Injection_C424'] = df['Production_Injection_C424'].clip(lower=0)
    
    # Use Production_Injection_C424 as power in watts
    print("Using Production_Injection_C424 as power (power_watts):")
    df['power_watts'] = df['Production_Injection_C424']
    df['power_watts'] = df['power_watts'].clip(lower=0)
    
    # Normalize power_watts for training stability
    df['power_watts_normalized'] = df['power_watts'] / MAX_POWER
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0  # Fractional hour for 15-min intervals
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['is_daytime'] = ((df['hour'] >= 7) & (df['hour'] <= 18)).astype(int)
    df['is_weekend'] = df['timestamp'].dt.weekday.isin([5, 6]).astype(int)
    
    # Calculate solar zenith angle
    def solar_zenith_angle(hour, day_of_year, latitude=LATITUDE):
        solar_time = hour + 0.17 * np.sin(4 * np.pi * (day_of_year - 80) / 373)
        declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        hour_angle = (solar_time - 12) * 15
        cos_zenith = (np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
                      np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle)))
        zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
        return zenith_angle
    df['solar_zenith_angle'] = [solar_zenith_angle(h, d) for h, d in zip(df['hour'], df['day_of_year'])]
    
    # Set power_watts to 0 when swflx=0
    df.loc[df['swflx'] == 0, 'power_watts'] = 0
    df.loc[df['swflx'] == 0, 'power_watts_normalized'] = 0
    
    # Check for non-zero power when swflx=0
    daytime_swflx_zero = df[(df['swflx'] == 0) & (df['is_daytime'] == 1) & (df['power_watts'] > 0)]
    if not daytime_swflx_zero.empty:
        print(f"Warning: Found {len(daytime_swflx_zero)} instances where swflx=0 but power_watts > 0 during daytime:")
        print(daytime_swflx_zero[['timestamp', 'swflx', 'power_watts', 'is_daytime']])
    
    # Add features (adjusted for 15-minute intervals)
    df['temp_swflx_interaction'] = df['temp'] * df['swflx']
    df['swflx_zenith_interaction'] = df['swflx'] * df['solar_zenith_angle']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['swflx_lag1'] = df['swflx'].shift(1).fillna(0)  # 1 period = 15 min
    df['swflx_lag24'] = df['swflx'].shift(96).fillna(0)  # 24 hours = 96 intervals
    df['temp_lag1'] = df['temp'].shift(1).fillna(df['temp'].mean())  # 1 period = 15 min
    df['swflx_rolling_mean'] = df['swflx'].rolling(window=12, min_periods=1).mean().fillna(0)  # 3 hours = 12 intervals
    df['swflx_rolling_mean_6'] = df['swflx'].rolling(window=24, min_periods=1).mean().fillna(0)  # 6 hours = 24 intervals
    df['swflx_rolling_std_6'] = df['swflx'].rolling(window=24, min_periods=1).std().fillna(0)  # 6 hours = 24 intervals
    df['swflx_rolling_max_6'] = df['swflx'].rolling(window=24, min_periods=1).max().fillna(0)  # 6 hours = 24 intervals
    df['temp_rolling_mean'] = df['temp'].rolling(window=12, min_periods=1).mean().fillna(df['temp'].mean())  # 3 hours = 12 intervals
    df['temp_rolling_std_6'] = df['temp'].rolling(window=24, min_periods=1).std().fillna(0)  # 6 hours = 24 intervals
    df['prod_inj_lag24'] = df['Production_Injection_C424'].shift(96).fillna(0)  # 24 hours = 96 intervals
    df['prod_inj_lag1'] = df['Production_Injection_C424'].shift(1).fillna(0)  # 1 period = 15 min
    df['swflx_peak_indicator'] = (df['swflx'] > df['swflx'].quantile(0.9)).astype(int)  # Flag high irradiance
    
    print("Data Summary Before Training:")
    print(df[['temp', 'swflx', 'Production_Injection_C424', 'power_watts']].describe())
    print(f"Max power_watts when swflx=0: {df[df['swflx'] == 0]['power_watts'].max()}")
    print(f"Total unique timestamps: {df['timestamp'].nunique()}")
    
    return df
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import DATE_COL, VALUE_COL, ANOMALY_Z_THRESHOLD

def detect_anomalies_zscore(daily_df, z_threshold=ANOMALY_Z_THRESHOLD):
    daily = daily_df.copy()
    
    mean = daily['daily_sales'].mean()
    std = daily['daily_sales'].std(ddof=0)
    
    if std == 0 or np.isnan(std):
        daily['z_score'] = 0.0
        return daily.iloc[0:0]
    
    daily['z_score'] = (daily['daily_sales'] - mean) / std
    
    anomalies = daily[abs(daily['z_score']) >= z_threshold].copy()
    
    # Label anomalies: High sales = Positive, Low sales = Warning
    anomalies['anomaly_type'] = anomalies['z_score'].apply(
        lambda x: 'High Sales' if x > 0 else 'Low Sales'
    )
    
    return anomalies

def detect_anomalies_isolation_forest(daily_df, contamination=0.05):
    daily = daily_df.copy()
    
    daily['day_of_week'] = daily[DATE_COL].dt.dayofweek
    daily['day_of_month'] = daily[DATE_COL].dt.day
    daily['month'] = daily[DATE_COL].dt.month
    # Use aggregated 'daily_sales' if present, otherwise fall back to configured VALUE_COL
    value_col = 'daily_sales' if 'daily_sales' in daily.columns else VALUE_COL

    features = [value_col, 'day_of_week', 'day_of_month', 'month']
    # Ensure there are no missing values in feature columns
    for col in features:
        if col not in daily.columns:
            daily[col] = 0
    X = daily[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(contamination=contamination, random_state=42)
    daily['anomaly'] = model.fit_predict(X_scaled)
    
    anomalies = daily[daily['anomaly'] == -1].copy()
    
    # Label high vs low anomalies
    mean_val = daily[value_col].mean()
    anomalies['anomaly_type'] = anomalies[value_col].apply(
        lambda x: 'High Sales' if x > mean_val else 'Low Sales'
    )
    
    return anomalies

def analyze_anomaly_patterns(anomalies_df):
    if len(anomalies_df) == 0:
        return "No anomalies detected."
    
    value_col = 'daily_sales' if 'daily_sales' in anomalies_df.columns else VALUE_COL

    patterns = {
        'total_anomalies': len(anomalies_df),
        'avg_anomaly_value': anomalies_df[value_col].mean(),
        'max_anomaly_value': anomalies_df[value_col].max(),
        'min_anomaly_value': anomalies_df[value_col].min(),
    }
    
    if 'day_of_week' in anomalies_df.columns:
        patterns['most_common_day'] = anomalies_df['day_of_week'].mode()[0] if len(anomalies_df) > 0 else None
    
    return patterns

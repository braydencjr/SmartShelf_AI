import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from config import DATE_COL, VALUE_COL, FORECAST_PERIODS


def preprocess_timeseries(ts_data):
    """
    Preprocess time series: handle missing values, remove outliers, ensure quality.
    """
    ts = ts_data.copy()
    
    # Forward fill then backward fill for missing values
    ts = ts.fillna(method='ffill').fillna(method='bfill')
    
    # Remove extreme outliers (beyond 3 std devs) and replace with median
    mean, std = ts.mean(), ts.std()
    if std > 0:
        outlier_mask = (ts - mean).abs() > 3 * std
        if outlier_mask.any():
            ts[outlier_mask] = ts.median()
    
    return ts


def is_stationary(ts_data, threshold=0.05):
    """Check if time series is stationary using Augmented Dickey-Fuller test."""
    try:
        result = adfuller(ts_data.dropna())
        return result[1] < threshold
    except:
        return False


def forecast_sales(daily_df, periods=FORECAST_PERIODS):
    """Optimized Exponential Smoothing with better initialization and bounds."""
    try:
        # Prepare data
        ts_data = daily_df.set_index(DATE_COL)['daily_sales']
        ts_data = preprocess_timeseries(ts_data)
        
        # Optimized Exponential Smoothing with better initialization
        model = ExponentialSmoothing(
            ts_data,
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        )
        
        fitted_model = model.fit(optimized=True)
        forecast = fitted_model.forecast(steps=periods)
        
        # Create forecast dataframe with dynamic confidence intervals
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        # Better confidence interval based on model residuals
        residuals = fitted_model.resid
        std_error = residuals.std()
        margin = 1.96 * std_error  # 95% confidence
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': forecast.values,
            'lower_bound': np.maximum(forecast.values - margin, 0),
            'upper_bound': forecast.values + margin
        })
        
        return forecast_df, fitted_model
        
    except Exception as e:
        print(f"Exponential Smoothing error: {e}")
        return None, None


def arima_forecast(daily_df, periods=FORECAST_PERIODS):
    """Improved ARIMA with auto stationarity detection and better parameters."""
    try:
        ts_data = daily_df.set_index(DATE_COL)['daily_sales']
        ts_data = preprocess_timeseries(ts_data)
        
        # Auto-tune ARIMA parameters based on data
        # Check stationarity to determine d parameter
        d = 0 if is_stationary(ts_data) else 1
        
        # Use more conservative parameters for better generalization
        model = ARIMA(ts_data, order=(2, d, 2))
        fitted_model = model.fit()
        
        forecast = fitted_model.get_forecast(steps=periods)
        forecast_df_result = forecast.conf_int()
        
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': np.maximum(forecast.predicted_mean.values, 0),
            'lower_bound': np.maximum(forecast_df_result.iloc[:, 0].values, 0),
            'upper_bound': np.maximum(forecast_df_result.iloc[:, 1].values, 0)
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"ARIMA forecasting error: {e}")
        return None


def sarimax_forecast(daily_df, periods=FORECAST_PERIODS):
    """Optimized SARIMAX with better hyperparameters and bounds."""
    try:
        # Support both daily aggregated frames (with 'daily_sales')
        # and raw/value frames (with VALUE_COL, e.g. 'TotalSales')
        value_col = 'daily_sales' if 'daily_sales' in daily_df.columns else VALUE_COL
        ts_data = daily_df.set_index(DATE_COL)[value_col]
        ts_data = preprocess_timeseries(ts_data)
        
        # Optimized SARIMAX with better hyperparameters
        model = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False, maxiter=500)
        forecast = fitted_model.get_forecast(steps=periods)

        forecast_dates = pd.date_range(start=daily_df[DATE_COL].max() + pd.Timedelta(days=1), periods=periods)
        
        # Extract confidence intervals
        conf = forecast.conf_int()
        if conf.shape[1] >= 2:
            lower = np.maximum(conf.iloc[:, 0].values, 0)
            upper = conf.iloc[:, 1].values
        else:
            # Fallback to residual-based bounds
            residuals = fitted_model.resid
            std_error = residuals.std()
            margin = 1.96 * std_error
            lower = np.maximum(forecast.predicted_mean.values - margin, 0)
            upper = forecast.predicted_mean.values + margin

        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': np.maximum(forecast.predicted_mean.values, 0),
            'lower_bound': lower,
            'upper_bound': upper
        })

        return forecast_df
        
    except Exception as e:
        print(f"SARIMAX forecasting error: {e}")
        return None


def ltm_forecast(daily_df, periods=FORECAST_PERIODS):
    """Improved Linear Trend Model with feature scaling and dynamic bounds."""
    try:
        # Use daily_sales if available, otherwise use configured VALUE_COL
        value_col = 'daily_sales' if 'daily_sales' in daily_df.columns else VALUE_COL
        ts_data = daily_df.set_index(DATE_COL)[value_col]
        ts_data = preprocess_timeseries(ts_data)
        
        # Add lag features for better linear regression
        X = np.arange(len(ts_data)).reshape(-1, 1)
        y = ts_data.values
        
        # Scale features for better numerical stability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Generate future X values and scale them
        future_X = np.arange(len(ts_data), len(ts_data) + periods).reshape(-1, 1)
        future_X_scaled = scaler.transform(future_X)
        forecast = model.predict(future_X_scaled)
        forecast = np.maximum(forecast, 0)

        forecast_dates = pd.date_range(start=daily_df[DATE_COL].max() + pd.Timedelta(days=1), periods=periods)
        
        # Dynamic confidence intervals based on residuals
        residuals = y - model.predict(X_scaled)
        std_error = residuals.std()
        margin = 1.96 * std_error
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': forecast,
            'lower_bound': np.maximum(forecast - margin, 0),
            'upper_bound': forecast + margin
        })
        return forecast_df
    except Exception as e:
        print(f"LTM forecasting error: {e}")
        return None


def ensemble_forecast(daily_df, periods=FORECAST_PERIODS):
    """
    Hybrid ensemble model combining SARIMAX, ARIMA, and Exponential Smoothing.
    Uses weighted average based on recent model performance.
    """
    try:
        value_col = 'daily_sales' if 'daily_sales' in daily_df.columns else VALUE_COL
        ts_data = daily_df.set_index(DATE_COL)[value_col]
        ts_data = preprocess_timeseries(ts_data)
        
        # Split data: train on 80%, validate on 20%
        split_idx = int(len(ts_data) * 0.8)
        train_data = ts_data.iloc[:split_idx]
        test_data = ts_data.iloc[split_idx:]
        
        forecasts = {}
        
        # SARIMAX forecast
        try:
            model_s = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False)
            fitted_s = model_s.fit(disp=False, maxiter=500)
            test_pred_s = fitted_s.forecast(steps=len(test_data))
            forecasts['sarimax_mape'] = np.mean(np.abs((test_data.values - test_pred_s) / np.maximum(test_data.values, 1))) * 100
            forecasts['sarimax'] = 1.0  # Will be normalized
        except:
            forecasts['sarimax_mape'] = 100
            forecasts['sarimax'] = 0.0
        
        # ARIMA forecast
        try:
            d = 0 if is_stationary(train_data) else 1
            model_a = ARIMA(train_data, order=(2, d, 2))
            fitted_a = model_a.fit()
            test_pred_a = fitted_a.forecast(steps=len(test_data))
            forecasts['arima_mape'] = np.mean(np.abs((test_data.values - test_pred_a) / np.maximum(test_data.values, 1))) * 100
            forecasts['arima'] = 1.0
        except:
            forecasts['arima_mape'] = 100
            forecasts['arima'] = 0.0
        
        # Exponential Smoothing forecast
        try:
            model_e = ExponentialSmoothing(train_data, seasonal_periods=7, trend='add', seasonal='add')
            fitted_e = model_e.fit(optimized=True)
            test_pred_e = fitted_e.forecast(steps=len(test_data))
            forecasts['exp_mape'] = np.mean(np.abs((test_data.values - test_pred_e) / np.maximum(test_data.values, 1))) * 100
            forecasts['exp'] = 1.0
        except:
            forecasts['exp_mape'] = 100
            forecasts['exp'] = 0.0
        
        # Calculate weights (inverse of MAPE, normalize)
        weights = {
            'sarimax': max(0, 100 - forecasts['sarimax_mape']) if forecasts['sarimax'] > 0 else 0,
            'arima': max(0, 100 - forecasts['arima_mape']) if forecasts['arima'] > 0 else 0,
            'exp': max(0, 100 - forecasts['exp_mape']) if forecasts['exp'] > 0 else 0
        }
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            weights = {'sarimax': 0.5, 'arima': 0.3, 'exp': 0.2}
        else:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Generate forecasts using full dataset
        forecasts_final = {}
        
        try:
            model_s_full = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False)
            fitted_s_full = model_s_full.fit(disp=False, maxiter=500)
            fore_s = fitted_s_full.get_forecast(steps=periods)
            forecasts_final['sarimax'] = np.maximum(fore_s.predicted_mean.values, 0)
        except:
            forecasts_final['sarimax'] = np.full(periods, ts_data.mean())
        
        try:
            d = 0 if is_stationary(ts_data) else 1
            model_a_full = ARIMA(ts_data, order=(2, d, 2))
            fitted_a_full = model_a_full.fit()
            fore_a = fitted_a_full.get_forecast(steps=periods)
            forecasts_final['arima'] = np.maximum(fore_a.predicted_mean.values, 0)
        except:
            forecasts_final['arima'] = np.full(periods, ts_data.mean())
        
        try:
            model_e_full = ExponentialSmoothing(ts_data, seasonal_periods=7, trend='add', seasonal='add')
            fitted_e_full = model_e_full.fit(optimized=True)
            fore_e = fitted_e_full.forecast(steps=periods)
            forecasts_final['exp'] = np.maximum(fore_e.values, 0)
        except:
            forecasts_final['exp'] = np.full(periods, ts_data.mean())
        
        # Weighted ensemble
        ensemble_pred = (
            weights['sarimax'] * forecasts_final['sarimax'] +
            weights['arima'] * forecasts_final['arima'] +
            weights['exp'] * forecasts_final['exp']
        )
        
        # Calculate confidence intervals
        std_error = np.std([
            forecasts_final['sarimax'] - ensemble_pred,
            forecasts_final['arima'] - ensemble_pred,
            forecasts_final['exp'] - ensemble_pred
        ])
        margin = 1.96 * std_error
        
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': ensemble_pred,
            'lower_bound': np.maximum(ensemble_pred - margin, 0),
            'upper_bound': ensemble_pred + margin,
            'sarimax_weight': weights['sarimax'],
            'arima_weight': weights['arima'],
            'exp_weight': weights['exp']
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"Ensemble forecasting error: {e}")
        return None


def calculate_forecast_accuracy(actual, predicted):
    """Calculate multiple accuracy metrics with edge case handling."""
    # Handle edge cases
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    if len(actual) == 0 or len(predicted) == 0:
        return {'MAPE': 0, 'RMSE': 0, 'MAE': 0}
    
    # Avoid division by zero in MAPE
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = 0
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    return {
        'MAPE': round(mape, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2)
    }

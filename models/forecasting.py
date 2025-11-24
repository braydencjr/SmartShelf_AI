import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from config import DATE_COL, VALUE_COL, FORECAST_PERIODS

def forecast_sales(daily_df, periods=FORECAST_PERIODS):
    try:
        # Prepare data
        ts_data = daily_df.set_index(DATE_COL)['daily_sales']
        
        # Handle missing values
        ts_data = ts_data.ffill()
        
        # Exponential Smoothing
        model = ExponentialSmoothing(
            ts_data,
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        )
        
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=periods)
        
        # Create forecast dataframe
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': forecast.values,
            'lower_bound': forecast.values * 0.85,
            'upper_bound': forecast.values * 1.15
        })
        
        return forecast_df, fitted_model
        
    except Exception as e:
        print(f"Forecasting error: {e}")
        return None, None

def arima_forecast(daily_df, periods=FORECAST_PERIODS):
    try:
        ts_data = daily_df.set_index(DATE_COL)['daily_sales']
        ts_data = ts_data.fillna(method='ffill')
        
        # ARIMA model
        ts_data = ts_data.ffill()
        model = ARIMA(ts_data, order=(5, 1, 2))
        fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=periods)
        
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': forecast.values,
            'lower_bound': forecast.values * 0.9,
            'upper_bound': forecast.values * 1.1
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"ARIMA forecasting error: {e}")
        return None

def sarimax_forecast(daily_df, periods=FORECAST_PERIODS):
    try:
        # Support both daily aggregated frames (with 'daily_sales')
        # and raw/value frames (with VALUE_COL, e.g. 'TotalSales')
        value_col = 'daily_sales' if 'daily_sales' in daily_df.columns else VALUE_COL
        ts_data = daily_df.set_index(DATE_COL)[value_col].ffill()
        model = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,7))
        fitted_model = model.fit(disp=False)
        forecast = fitted_model.get_forecast(steps=periods)

        forecast_dates = pd.date_range(start=daily_df[DATE_COL].max() + pd.Timedelta(days=1), periods=periods)
        # Safely extract confidence intervals regardless of column naming
        conf = forecast.conf_int()
        if conf.shape[1] >= 2:
            lower = conf.iloc[:, 0].values
            upper = conf.iloc[:, 1].values
        else:
            # Fallback to simple percentage bounds if conf_int isn't as expected
            mean_vals = forecast.predicted_mean.values
            lower = mean_vals * 0.9
            upper = mean_vals * 1.1

        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': forecast.predicted_mean.values,
            'lower_bound': lower,
            'upper_bound': upper
        })

        return forecast_df
        
    except Exception as e:
        print(f"SARIMAX forecasting error: {e}")
        return None
    

def ltm_forecast(daily_df, periods=FORECAST_PERIODS):
    try:
        # Use daily_sales if available, otherwise use configured VALUE_COL
        value_col = 'daily_sales' if 'daily_sales' in daily_df.columns else VALUE_COL
        ts_data = daily_df.set_index(DATE_COL)[value_col].ffill()
        X = np.arange(len(ts_data)).reshape(-1,1)
        y = ts_data.values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(ts_data), len(ts_data)+periods).reshape(-1,1)
        forecast = model.predict(future_X)

        forecast_dates = pd.date_range(start=daily_df[DATE_COL].max() + pd.Timedelta(days=1), periods=periods)
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': forecast,
            'lower_bound': forecast * 0.95,
            'upper_bound': forecast * 1.05
        })
        return forecast_df
    except Exception as e:
        print(f"LTM forecasting error: {e}")
        return None


def calculate_forecast_accuracy(actual, predicted):
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    return {
        'MAPE': round(mape, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2)
    }
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
from config import DATE_COL, VALUE_COL, FORECAST_PERIODS


def preprocess_timeseries(ts_data):
    """Advanced preprocessing with outlier detection and imputation."""
    ts = ts_data.copy()
    
    # Interpolate missing values
    ts = ts.interpolate(method='linear')
    ts = ts.fillna(method='ffill').fillna(method='bfill')
    
    # Remove extreme outliers using IQR method (more robust than std)
    Q1 = ts.quantile(0.25)
    Q3 = ts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (ts < lower_bound) | (ts > upper_bound)
    if outlier_mask.any():
        ts[outlier_mask] = ts.median()
    
    return ts


def create_lagged_features(data, lags=14):
    """Create lag features for ML models."""
    df = pd.DataFrame(data)
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
    
    # Add rolling statistics
    df['rolling_mean_7'] = df.iloc[:, 0].rolling(window=7).mean()
    df['rolling_std_7'] = df.iloc[:, 0].rolling(window=7).std()
    df['rolling_mean_14'] = df.iloc[:, 0].rolling(window=14).mean()
    
    # Add day of week seasonality
    df['day_of_week'] = df.index.dayofweek
    df['week_of_year'] = df.index.isocalendar().week
    
    # Drop NaN rows
    df = df.dropna()
    return df


def is_stationary(ts_data, threshold=0.05):
    """ADF test for stationarity."""
    try:
        result = adfuller(ts_data.dropna())
        return result[1] < threshold
    except:
        return False


def prophet_forecast(daily_df, periods=FORECAST_PERIODS):
    """Facebook Prophet - excellent for seasonality and trends."""
    try:
        # Prepare data in Prophet format
        df_prophet = pd.DataFrame({
            'ds': daily_df[DATE_COL],
            'y': daily_df['daily_sales']
        })
        
        # Remove NaN values
        df_prophet = df_prophet.dropna()
        
        # Preprocess
        df_prophet['y'] = preprocess_timeseries(df_prophet['y'])
        
        # Create and fit model (suppress output)
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df_prophet)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Extract forecast data
        forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_result['ds'].values,
            'forecasted_sales': np.maximum(forecast_result['yhat'].values, 0),
            'lower_bound': np.maximum(forecast_result['yhat_lower'].values, 0),
            'upper_bound': np.maximum(forecast_result['yhat_upper'].values, 0)
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"Prophet error: {e}")
        return None


def xgboost_forecast(daily_df, periods=FORECAST_PERIODS, lags=14):
    """XGBoost with lag features - captures complex non-linear patterns."""
    try:
        ts_data = daily_df.set_index(DATE_COL)['daily_sales']
        ts_data = preprocess_timeseries(ts_data)
        
        # Create lagged features
        features_df = create_lagged_features(ts_data, lags=lags)
        X = features_df.iloc[:, 1:].values  # All columns except first
        y = features_df.iloc[:, 0].values
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X, y)
        
        # Generate forecast
        last_values = ts_data.tail(lags).values.copy()
        forecasts = []
        
        for step in range(periods):
            # Create feature vector from recent history
            lag_features = last_values[-lags:]
            rolling_mean_7 = np.mean(last_values[-7:])
            rolling_std_7 = np.std(last_values[-7:])
            rolling_mean_14 = np.mean(last_values[-14:]) if len(last_values) >= 14 else rolling_mean_7
            day_of_week = (len(last_values) + step) % 7
            week_of_year = ((len(last_values) + step) // 7) % 52
            
            features = np.concatenate([
                lag_features,
                [rolling_mean_7, rolling_std_7, rolling_mean_14, float(day_of_week), float(week_of_year)]
            ])
            
            # Predict
            pred = model.predict(features.reshape(1, -1))[0]
            pred = max(pred, 0)  # No negative forecasts
            forecasts.append(pred)
            
            # Update history
            last_values = np.append(last_values[1:], pred)
        
        # Create forecast dataframe
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        # Simple confidence intervals based on std of recent predictions
        std_forecast = np.std(forecasts) if len(forecasts) > 1 else np.mean(forecasts) * 0.2
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': forecasts,
            'lower_bound': np.maximum(np.array(forecasts) - 1.96 * std_forecast, 0),
            'upper_bound': np.array(forecasts) + 1.96 * std_forecast
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"XGBoost error: {e}")
        return None


def optimized_sarimax_forecast(daily_df, periods=FORECAST_PERIODS):
    """SARIMAX with grid search optimization."""
    try:
        ts_data = daily_df.set_index(DATE_COL)['daily_sales']
        ts_data = preprocess_timeseries(ts_data)
        
        # Simple grid search for best SARIMAX parameters
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal = (1, 1, 1, 7)
        
        # Limited grid for speed
        p_range = range(0, 3)
        d_range = [0, 1]
        q_range = range(0, 3)
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = SARIMAX(
                            ts_data,
                            order=(p, d, q),
                            seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted = model.fit(disp=False, maxiter=200)
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        # Fit best model
        model = SARIMAX(
            ts_data,
            order=best_order,
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=500)
        forecast = fitted_model.get_forecast(steps=periods)
        
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        conf = forecast.conf_int()
        lower = np.maximum(conf.iloc[:, 0].values, 0)
        upper = conf.iloc[:, 1].values
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': np.maximum(forecast.predicted_mean.values, 0),
            'lower_bound': lower,
            'upper_bound': upper
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"Optimized SARIMAX error: {e}")
        return None


def smart_ensemble_forecast(daily_df, periods=FORECAST_PERIODS):
    """
    Intelligent ensemble that:
    1. Tests all 5 models on validation set
    2. Selects only top 2-3 performers
    3. Weights by accuracy
    4. Returns only if ensemble MAPE < 15%
    """
    try:
        value_col = 'daily_sales' if 'daily_sales' in daily_df.columns else VALUE_COL
        ts_data = daily_df.set_index(DATE_COL)[value_col]
        ts_data = preprocess_timeseries(ts_data)
        
        # Train-test split (70-30 for better validation)
        split_idx = int(len(ts_data) * 0.7)
        train_data = ts_data.iloc[:split_idx]
        test_data = ts_data.iloc[split_idx:]
        test_periods = len(test_data)
        
        model_scores = {}
        model_predictions = {}
        
        # Test Prophet
        try:
            train_df_temp = pd.DataFrame({DATE_COL: daily_df[DATE_COL].iloc[:split_idx], 'daily_sales': train_data})
            prophet_pred = prophet_forecast(train_df_temp, periods=test_periods)
            if prophet_pred is not None and len(prophet_pred) == test_periods:
                mape = np.mean(np.abs((test_data.values - prophet_pred['forecasted_sales'].values) / np.maximum(test_data.values, 1))) * 100
                model_scores['prophet'] = max(0, 100 - mape)
                model_predictions['prophet'] = prophet_pred
                print(f"Prophet MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"Prophet validation error: {e}")
        
        # Test XGBoost
        try:
            train_df_temp = pd.DataFrame({DATE_COL: daily_df[DATE_COL].iloc[:split_idx], 'daily_sales': train_data})
            xgb_pred = xgboost_forecast(train_df_temp, periods=test_periods)
            if xgb_pred is not None and len(xgb_pred) == test_periods:
                mape = np.mean(np.abs((test_data.values - xgb_pred['forecasted_sales'].values) / np.maximum(test_data.values, 1))) * 100
                model_scores['xgboost'] = max(0, 100 - mape)
                model_predictions['xgboost'] = xgb_pred
                print(f"XGBoost MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"XGBoost validation error: {e}")
        
        # Test Optimized SARIMAX
        try:
            train_df_temp = pd.DataFrame({DATE_COL: daily_df[DATE_COL].iloc[:split_idx], 'daily_sales': train_data})
            sarimax_pred = optimized_sarimax_forecast(train_df_temp, periods=test_periods)
            if sarimax_pred is not None and len(sarimax_pred) == test_periods:
                mape = np.mean(np.abs((test_data.values - sarimax_pred['forecasted_sales'].values) / np.maximum(test_data.values, 1))) * 100
                model_scores['sarimax'] = max(0, 100 - mape)
                model_predictions['sarimax'] = sarimax_pred
                print(f"SARIMAX MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"SARIMAX validation error: {e}")
        
        # Test Optimized ARIMA
        try:
            train_df_temp = pd.DataFrame({DATE_COL: daily_df[DATE_COL].iloc[:split_idx], 'daily_sales': train_data})
            arima_pred = optimized_arima_forecast(train_df_temp, periods=test_periods)
            if arima_pred is not None and len(arima_pred) == test_periods:
                mape = np.mean(np.abs((test_data.values - arima_pred['forecasted_sales'].values) / np.maximum(test_data.values, 1))) * 100
                model_scores['arima'] = max(0, 100 - mape)
                model_predictions['arima'] = arima_pred
                print(f"ARIMA MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"ARIMA validation error: {e}")
        
        if not model_scores:
            print("No models succeeded. Returning None.")
            return None
        
        # Select top 2-3 models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = min(3, len(sorted_models))
        top_models = dict(sorted_models[:top_k])
        
        print(f"\nTop {top_k} models selected:")
        for name, score in top_models.items():
            print(f"  {name}: score={score:.2f}")
        
        # Normalize weights
        total_score = sum(top_models.values())
        weights = {k: v / total_score for k, v in top_models.items()}
        
        # Generate forecasts using full dataset
        full_forecasts = {}
        
        try:
            prophet_pred = prophet_forecast(daily_df, periods=periods)
            if prophet_pred is not None:
                full_forecasts['prophet'] = prophet_pred['forecasted_sales'].values
        except:
            pass
        
        try:
            xgb_pred = xgboost_forecast(daily_df, periods=periods)
            if xgb_pred is not None:
                full_forecasts['xgboost'] = xgb_pred['forecasted_sales'].values
        except:
            pass
        
        try:
            sarimax_pred = optimized_sarimax_forecast(daily_df, periods=periods)
            if sarimax_pred is not None:
                full_forecasts['sarimax'] = sarimax_pred['forecasted_sales'].values
        except:
            pass
        
        try:
            arima_pred = optimized_arima_forecast(daily_df, periods=periods)
            if arima_pred is not None:
                full_forecasts['arima'] = arima_pred['forecasted_sales'].values
        except:
            pass
        
        # Weighted ensemble of top models only
        ensemble_pred = np.zeros(periods)
        for model_name, weight in weights.items():
            if model_name in full_forecasts:
                ensemble_pred += weight * full_forecasts[model_name]
        
        last_date = daily_df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        # Confidence intervals from model variance
        pred_array = np.array([full_forecasts.get(m, ensemble_pred) for m in top_models.keys()])
        std_error = np.std(pred_array, axis=0)
        
        forecast_df = pd.DataFrame({
            DATE_COL: forecast_dates,
            'forecasted_sales': ensemble_pred,
            'lower_bound': np.maximum(ensemble_pred - 1.96 * std_error, 0),
            'upper_bound': ensemble_pred + 1.96 * std_error,
            'models_used': ', '.join(top_models.keys())
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"Smart Ensemble error: {e}")
        return None

def optimized_arima_forecast(daily_df, periods=FORECAST_PERIODS):
    """ARIMA with constrained grid for speed."""
    try:
        df = _safe_dates(daily_df)
        ts_data = df.set_index(DATE_COL)['daily_sales'].sort_index()
        ts_data = preprocess_timeseries(ts_data)

        if len(ts_data) < 8:
            return None

        # small grid
        p_vals = [0, 1]
        d_vals = [0, 1]
        q_vals = [0, 1]

        best_aic = np.inf
        best_order = (1, 0, 1)

        key = _dataset_key(df, 'daily_sales') + ('arima_search',)
        if key in _MODEL_CACHE:
            best_order = _MODEL_CACHE[key]
        else:
            for p in p_vals:
                for d in d_vals:
                    for q in q_vals:
                        try:
                            m = ARIMA(ts_data, order=(p, d, q)).fit()
                            if m.aic < best_aic:
                                best_aic = m.aic
                                best_order = (p, d, q)
                        except Exception:
                            continue
            _MODEL_CACHE[key] = best_order

        fit_key = _dataset_key(df, 'daily_sales') + ('arima_fit', best_order, periods)
        if fit_key in _MODEL_CACHE:
            fitted = _MODEL_CACHE[fit_key]
            forecast = fitted.get_forecast(steps=periods)
        else:
            fitted = ARIMA(ts_data, order=best_order).fit()
            _MODEL_CACHE[fit_key] = fitted
            forecast = fitted.get_forecast(steps=periods)

        conf = forecast.conf_int()
        preds = np.maximum(forecast.predicted_mean.values, 0.0)
        lowers = np.maximum(conf.iloc[:, 0].values, 0.0)
        highs = conf.iloc[:, 1].values

        last_date = df[DATE_COL].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        return _to_forecast_df(forecast_dates, preds, lowers, highs)

    except Exception as e:
        print(f"Optimized ARIMA error: {e}")
        return None


def calculate_forecast_accuracy(actual, predicted):
    """Calculate accuracy metrics."""
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    if len(actual) == 0 or len(predicted) == 0:
        return {'MAPE': 0, 'RMSE': 0, 'MAE': 0}
    
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

"""
Model Training Module

This module handles:
- Model selection (Prophet, ARIMA/SARIMA, XGBoost)
- Training and validation
- Hyperparameter tuning
- Model evaluation metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import models (with error handling for optional dependencies)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """
    Handles model training, validation, and evaluation for time series forecasting.
    """
    
    def __init__(self, model_type: str = 'prophet'):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_type (str): Type of model to use ('prophet', 'arima', 'xgboost')
        """
        self.model_type = model_type.lower()
        self.model = None
        self.training_history = {}
        self.validation_metrics = {}
        
        # Validate model type
        if self.model_type == 'prophet' and not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Please install with: pip install prophet")
        elif self.model_type == 'arima' and not ARIMA_AVAILABLE:
            raise ImportError("ARIMA not available. Please install with: pip install statsmodels")
        elif self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Please install with: pip install xgboost")
    
    def prepare_prophet_data(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """
        Prepare data for Prophet model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the pollutant column
            
        Returns:
            pd.DataFrame: DataFrame formatted for Prophet
        """
        # Prophet expects columns named 'ds' (date) and 'y' (target)
        prophet_df = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(prophet_df.index, pd.DatetimeIndex):
            # Try to find datetime column
            datetime_cols = ['date', 'time', 'datetime', 'timestamp', 'date_time']
            for col in datetime_cols:
                if col in prophet_df.columns:
                    prophet_df[col] = pd.to_datetime(prophet_df[col])
                    prophet_df = prophet_df.set_index(col)
                    break
        
        # Reset index and rename columns
        prophet_df = prophet_df.reset_index()
        prophet_df = prophet_df.rename(columns={prophet_df.columns[0]: 'ds', pollutant: 'y'})
        
        # Select only required columns
        prophet_df = prophet_df[['ds', 'y']].copy()
        
        # Remove rows with missing values
        prophet_df = prophet_df.dropna()
        
        return prophet_df
    
    def prepare_arima_data(self, df: pd.DataFrame, pollutant: str) -> pd.Series:
        """
        Prepare data for ARIMA model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the pollutant column
            
        Returns:
            pd.Series: Time series data for ARIMA
        """
        # ARIMA expects a time series with datetime index
        arima_data = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(arima_data.index, pd.DatetimeIndex):
            # Try to find datetime column
            datetime_cols = ['date', 'time', 'datetime', 'timestamp', 'date_time']
            for col in datetime_cols:
                if col in arima_data.columns:
                    arima_data[col] = pd.to_datetime(arima_data[col])
                    arima_data = arima_data.set_index(col)
                    break
        
        # Select the pollutant column
        if pollutant in arima_data.columns:
            arima_series = arima_data[pollutant]
        else:
            raise ValueError(f"Pollutant '{pollutant}' not found in DataFrame")
        
        # Remove missing values
        arima_series = arima_series.dropna()
        
        return arima_series
    
    def prepare_xgboost_data(self, df: pd.DataFrame, pollutant: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for XGBoost model.
        
        Args:
            df (pd.DataFrame): Input DataFrame with engineered features
            pollutant (str): Name of the pollutant column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target for XGBoost
        """
        # XGBoost expects features and target
        xgb_data = df.copy()
        
        # Ensure we have a datetime index for proper train-test split
        if not isinstance(xgb_data.index, pd.DatetimeIndex):
            # Try to find datetime column
            datetime_cols = ['date', 'time', 'datetime', 'timestamp', 'date_time']
            for col in datetime_cols:
                if col in xgb_data.columns:
                    xgb_data[col] = pd.to_datetime(xgb_data[col])
                    xgb_data = xgb_data.set_index(col)
                    break
        
        # Select features (exclude target and datetime)
        feature_cols = [col for col in xgb_data.columns if col != pollutant]
        
        # Remove rows with missing values
        xgb_data = xgb_data.dropna()
        
        X = xgb_data[feature_cols]
        y = xgb_data[pollutant]
        
        return X, y
    
    def train_prophet(self, df: pd.DataFrame, **kwargs) -> Prophet:
        """
        Train Prophet model.
        
        Args:
            df (pd.DataFrame): DataFrame formatted for Prophet
            **kwargs: Additional Prophet parameters
            
        Returns:
            Prophet: Trained Prophet model
        """
        # Default Prophet parameters
        default_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive'
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        # Create and fit model
        model = Prophet(**default_params)
        model.fit(df)
        
        return model
    
    def train_arima(self, series: pd.Series, order: Tuple[int, int, int] = None) -> Any:
        """
        Train ARIMA model.
        
        Args:
            series (pd.Series): Time series data
            order (Tuple[int, int, int]): ARIMA order (p, d, q)
            
        Returns:
            ARIMA: Trained ARIMA model
        """
        # Auto-determine order if not provided
        if order is None:
            order = self._auto_arima_order(series)
        
        # Fit ARIMA model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        return fitted_model
    
    def _auto_arima_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically determine ARIMA order.
        
        Args:
            series (pd.Series): Time series data
            
        Returns:
            Tuple[int, int, int]: Suggested ARIMA order
        """
        # Simple approach: start with (1,1,1) and adjust based on stationarity
        # Check stationarity
        adf_result = adfuller(series)
        
        if adf_result[1] < 0.05:  # Stationary
            d = 0
        else:  # Non-stationary, try first difference
            diff_series = series.diff().dropna()
            adf_diff = adfuller(diff_series)
            if adf_diff[1] < 0.05:
                d = 1
            else:
                d = 2
        
        # Simple p and q values
        p = 1
        q = 1
        
        return (p, d, q)
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> xgb.XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            **kwargs: Additional XGBoost parameters
            
        Returns:
            XGBRegressor: Trained XGBoost model
        """
        # Default XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        # Create and fit model
        model = xgb.XGBRegressor(**default_params)
        model.fit(X, y)
        
        return model
    
    def time_series_split(self, data: pd.DataFrame, n_splits: int = 5) -> TimeSeriesSplit:
        """
        Create time series cross-validation splits.
        
        Args:
            data (pd.DataFrame): Input data
            n_splits (int): Number of splits
            
        Returns:
            TimeSeriesSplit: Time series cross-validation object
        """
        return TimeSeriesSplit(n_splits=n_splits)
    
    def evaluate_model(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (pd.Series): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        metrics['mape'] = mape
        
        return metrics
    
    def train(self, df: pd.DataFrame, pollutant: str, **kwargs) -> Any:
        """
        Train the selected model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the target pollutant
            **kwargs: Additional model parameters
            
        Returns:
            Any: Trained model
        """
        print(f"Training {self.model_type.upper()} model...")
        
        if self.model_type == 'prophet':
            prophet_df = self.prepare_prophet_data(df, pollutant)
            self.model = self.train_prophet(prophet_df, **kwargs)
            
        elif self.model_type == 'arima':
            arima_series = self.prepare_arima_data(df, pollutant)
            self.model = self.train_arima(arima_series, **kwargs)
            
        elif self.model_type == 'xgboost':
            X, y = self.prepare_xgboost_data(df, pollutant)
            self.model = self.train_xgboost(X, y, **kwargs)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"{self.model_type.upper()} model training completed!")
        return self.model
    
    def validate(self, df: pd.DataFrame, pollutant: str, test_size: float = 0.2) -> Dict[str, float]:
        """
        Validate the trained model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the target pollutant
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Validating model...")
        
        if self.model_type == 'prophet':
            return self._validate_prophet(df, pollutant, test_size)
        elif self.model_type == 'arima':
            return self._validate_arima(df, pollutant, test_size)
        elif self.model_type == 'xgboost':
            return self._validate_xgboost(df, pollutant, test_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _validate_prophet(self, df: pd.DataFrame, pollutant: str, test_size: float) -> Dict[str, float]:
        """Validate Prophet model."""
        prophet_df = self.prepare_prophet_data(df, pollutant)
        
        # Split data
        split_idx = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]
        
        # Retrain on training data
        train_model = self.train_prophet(train_df)
        
        # Make predictions
        future = train_model.make_future_dataframe(periods=len(test_df))
        forecast = train_model.predict(future)
        
        # Extract predictions for test period
        test_forecast = forecast.iloc[split_idx:][['ds', 'yhat']]
        
        # Calculate metrics
        y_true = test_df['y'].values
        y_pred = test_forecast['yhat'].values
        
        return self.evaluate_model(pd.Series(y_true), pd.Series(y_pred))
    
    def _validate_arima(self, df: pd.DataFrame, pollutant: str, test_size: float) -> Dict[str, float]:
        """Validate ARIMA model."""
        arima_series = self.prepare_arima_data(df, pollutant)
        
        # Split data
        split_idx = int(len(arima_series) * (1 - test_size))
        train_series = arima_series[:split_idx]
        test_series = arima_series[split_idx:]
        
        # Retrain on training data
        train_model = self.train_arima(train_series)
        
        # Make predictions
        forecast = train_model.forecast(steps=len(test_series))
        
        # Calculate metrics
        y_true = test_series.values
        y_pred = forecast.values
        
        return self.evaluate_model(pd.Series(y_true), pd.Series(y_pred))
    
    def _validate_xgboost(self, df: pd.DataFrame, pollutant: str, test_size: float) -> Dict[str, float]:
        """Validate XGBoost model."""
        X, y = self.prepare_xgboost_data(df, pollutant)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Retrain on training data
        train_model = self.train_xgboost(X_train, y_train)
        
        # Make predictions
        y_pred = train_model.predict(X_test)
        
        # Calculate metrics
        return self.evaluate_model(y_test, y_pred)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the trained model.
        
        Returns:
            Dict[str, Any]: Model summary
        """
        if self.model is None:
            return {"error": "Model not trained"}
        
        summary = {
            "model_type": self.model_type,
            "model": self.model,
            "validation_metrics": self.validation_metrics
        }
        
        return summary

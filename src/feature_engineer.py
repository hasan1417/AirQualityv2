"""
Feature Engineering Module

This module creates features for time series forecasting:
- Lagged values
- Rolling statistics
- Time-based features
- Seasonal patterns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Creates engineered features for time series forecasting.
    """
    
    def __init__(self, max_lags: int = 7, rolling_windows: List[int] = None):
        """
        Initialize the FeatureEngineer.
        
        Args:
            max_lags (int): Maximum number of lagged features to create
            rolling_windows (List[int]): List of rolling window sizes for statistics
        """
        self.max_lags = max_lags
        self.rolling_windows = rolling_windows or [3, 7, 14, 30]
        self.feature_names = []
        
    def create_lagged_features(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """
        Create lagged features for the target pollutant.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            pollutant (str): Name of the pollutant column
            
        Returns:
            pd.DataFrame: DataFrame with lagged features
        """
        df_copy = df.copy()
        
        # Create lagged features
        for lag in range(1, self.max_lags + 1):
            lag_col = f"{pollutant}_lag_{lag}"
            df_copy[lag_col] = df_copy[pollutant].shift(lag)
            self.feature_names.append(lag_col)
            
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the pollutant column
            
        Returns:
            pd.DataFrame: DataFrame with rolling features
        """
        df_copy = df.copy()
        
        for window in self.rolling_windows:
            if window <= len(df):
                # Rolling mean
                mean_col = f"{pollutant}_rolling_mean_{window}"
                df_copy[mean_col] = df_copy[pollutant].rolling(window=window, min_periods=1).mean()
                self.feature_names.append(mean_col)
                
                # Rolling std
                std_col = f"{pollutant}_rolling_std_{window}"
                df_copy[std_col] = df_copy[pollutant].rolling(window=window, min_periods=1).std()
                self.feature_names.append(std_col)
                
                # Rolling min
                min_col = f"{pollutant}_rolling_min_{window}"
                df_copy[min_col] = df_copy[pollutant].rolling(window=window, min_periods=1).min()
                self.feature_names.append(min_col)
                
                # Rolling max
                max_col = f"{pollutant}_rolling_max_{window}"
                df_copy[pollutant].rolling(window=window, min_periods=1).max()
                self.feature_names.append(max_col)
                
        return df_copy
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime index.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            
        Returns:
            pd.DataFrame: DataFrame with time features
        """
        df_copy = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            # Try to find datetime column
            datetime_cols = ['date', 'time', 'datetime', 'timestamp', 'date_time']
            for col in datetime_cols:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_datetime(df_copy[col])
                    df_copy = df_copy.set_index(col)
                    break
            else:
                print("Warning: No datetime column found. Time features cannot be created.")
                return df_copy
        
        # Extract time components
        df_copy['day_of_week'] = df_copy.index.dayofweek
        df_copy['day_of_year'] = df_copy.index.dayofyear
        df_copy['month'] = df_copy.index.month
        df_copy['quarter'] = df_copy.index.quarter
        df_copy['year'] = df_copy.index.year
        df_copy['is_weekend'] = df_copy.index.dayofweek.isin([5, 6]).astype(int)
        
        # Cyclical encoding for periodic features
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        df_copy['day_of_year_sin'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365)
        df_copy['day_of_year_cos'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365)
        
        # Add time features to feature names
        time_features = ['day_of_week', 'day_of_year', 'month', 'quarter', 'year', 'is_weekend',
                        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                        'day_of_year_sin', 'day_of_year_cos']
        self.feature_names.extend(time_features)
        
        return df_copy
    
    def create_difference_features(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """
        Create difference features (first and second order differences).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the pollutant column
            
        Returns:
            pd.DataFrame: DataFrame with difference features
        """
        df_copy = df.copy()
        
        # First order difference
        diff1_col = f"{pollutant}_diff_1"
        df_copy[diff1_col] = df_copy[pollutant].diff()
        self.feature_names.append(diff1_col)
        
        # Second order difference
        diff2_col = f"{pollutant}_diff_2"
        df_copy[diff2_col] = df_copy[pollutant].diff().diff()
        self.feature_names.append(diff2_col)
        
        return df_copy
    
    def create_seasonal_features(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """
        Create seasonal decomposition features.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            pollutant (str): Name of the pollutant column
            
        Returns:
            pd.DataFrame: DataFrame with seasonal features
        """
        df_copy = df.copy()
        
        # Simple seasonal patterns (assuming daily data)
        # Weekly seasonality
        df_copy[f"{pollutant}_weekly_pattern"] = df_copy[pollutant].rolling(window=7, min_periods=1).mean()
        
        # Monthly seasonality (approximate)
        df_copy[f"{pollutant}_monthly_pattern"] = df_copy[pollutant].rolling(window=30, min_periods=1).mean()
        
        # Quarterly seasonality
        df_copy[f"{pollutant}_quarterly_pattern"] = df_copy[pollutant].rolling(window=90, min_periods=1).mean()
        
        seasonal_features = [f"{pollutant}_weekly_pattern", f"{pollutant}_monthly_pattern", 
                           f"{pollutant}_quarterly_pattern"]
        self.feature_names.extend(seasonal_features)
        
        return df_copy
    
    def handle_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in engineered features.
        
        Args:
            df (pd.DataFrame): Input DataFrame with engineered features
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_copy = df.copy()
        
        # Forward fill for lagged features
        lag_cols = [col for col in df_copy.columns if 'lag_' in col]
        df_copy[lag_cols] = df_copy[lag_cols].fillna(method='ffill')
        
        # Forward fill for rolling features
        rolling_cols = [col for col in df_copy.columns if 'rolling_' in col]
        df_copy[rolling_cols] = df_copy[rolling_cols].fillna(method='ffill')
        
        # Forward fill for difference features
        diff_cols = [col for col in df_copy.columns if 'diff_' in col]
        df_copy[diff_cols] = df_copy[diff_cols].fillna(0)  # Differences start at 0
        
        # Forward fill for seasonal features
        seasonal_cols = [col for col in df_copy.columns if 'pattern' in col]
        df_copy[seasonal_cols] = df_copy[seasonal_cols].fillna(method='ffill')
        
        # Forward fill any remaining NaNs
        df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
        
        return df_copy
    
    def create_features(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the target pollutant column
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        print("Creating lagged features...")
        df_features = self.create_lagged_features(df, pollutant)
        
        print("Creating rolling statistics...")
        df_features = self.create_rolling_features(df_features, pollutant)
        
        print("Creating time-based features...")
        df_features = self.create_time_features(df_features)
        
        print("Creating difference features...")
        df_features = self.create_difference_features(df_features, pollutant)
        
        print("Creating seasonal features...")
        df_features = self.create_seasonal_features(df_features, pollutant)
        
        print("Handling missing values in features...")
        df_features = self.handle_missing_features(df_features)
        
        print(f"Created {len(self.feature_names)} engineered features")
        print(f"Feature names: {self.feature_names[:10]}...")  # Show first 10
        
        return df_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of created feature names.
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names.copy()
    
    def select_features(self, df: pd.DataFrame, feature_subset: List[str] = None) -> pd.DataFrame:
        """
        Select a subset of features for modeling.
        
        Args:
            df (pd.DataFrame): Input DataFrame with all features
            feature_subset (List[str]): List of features to select. If None, selects all.
            
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        if feature_subset is None:
            feature_subset = self.feature_names
            
        # Always include the target variable and datetime index
        essential_cols = [col for col in df.columns if col not in self.feature_names]
        selected_features = [col for col in feature_subset if col in df.columns]
        
        selected_cols = essential_cols + selected_features
        return df[selected_cols]
    
    def create_forecast_features(self, df: pd.DataFrame, pollutant: str, forecast_horizon: int = 7) -> pd.DataFrame:
        """
        Create features for forecasting future values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pollutant (str): Name of the pollutant column
            forecast_horizon (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: DataFrame with features for forecasting
        """
        # This method would be used when creating features for future dates
        # For now, we'll create a simple extension
        df_copy = df.copy()
        
        # Get the last date in the data
        last_date = df_copy.index.max()
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=forecast_horizon, freq='D')
        
        # Create a DataFrame for future dates
        future_df = pd.DataFrame(index=future_dates)
        
        # Add time features for future dates
        future_df = self.create_time_features(future_df)
        
        # Combine with original data
        combined_df = pd.concat([df_copy, future_df])
        
        return combined_df

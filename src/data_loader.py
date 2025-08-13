"""
Data Loading and Preprocessing Module

This module handles:
- Loading CSV files for each city
- Concatenating data with city identification
- DateTime parsing and indexing
- Missing value handling
- Data resampling to daily frequency
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Handles data loading and preprocessing for air quality forecasting.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Directory containing CSV files
        """
        self.data_dir = data_dir
        self.combined_data = None
        self.pollutants = None
        
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the data directory.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping city names to DataFrames
        """
        city_data = {}
        
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory '{self.data_dir}' not found.")
            return city_data
            
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in '{self.data_dir}' directory.")
            return city_data
            
        for csv_file in csv_files:
            city_name = csv_file.replace('.csv', '').replace('_', ' ').title()
            file_path = os.path.join(self.data_dir, csv_file)
            
            try:
                df = pd.read_csv(file_path)
                city_data[city_name] = df
                print(f"Loaded {city_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                
        return city_data
    
    def identify_pollutants(self, city_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Identify available pollutant columns across all cities.
        
        Args:
            city_data (Dict[str, pd.DataFrame]): City data dictionary
            
        Returns:
            List[str]: List of available pollutant columns
        """
        all_columns = set()
        for df in city_data.values():
            all_columns.update(df.columns)
            
        # Common pollutant column names
        pollutant_keywords = ['pm', 'no2', 'so2', 'co', 'o3', 'aqi', 'pollutant']
        pollutants = []
        
        for col in all_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in pollutant_keywords):
                pollutants.append(col)
                
        # Also check for numeric columns that might be pollutants
        for city, df in city_data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in pollutants and col.lower() not in ['year', 'month', 'day', 'hour', 'minute']:
                    pollutants.append(col)
                    
        return list(set(pollutants))
    
    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse datetime columns and set as index.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with datetime index
        """
        # Common datetime column names
        datetime_cols = ['date', 'time', 'datetime', 'timestamp', 'date_time']
        
        for col in datetime_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    return df
                except:
                    continue
                    
        # If no standard datetime column found, try to infer
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    return df
                except:
                    continue
                    
        print("Warning: No datetime column found. Please ensure your data has a datetime column.")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            method (str): Method for handling missing values ('interpolate', 'ffill', 'bfill', 'drop')
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        if method == 'interpolate':
            df = df.interpolate(method='time' if df.index.dtype == 'datetime64[ns]' else 'linear')
        elif method == 'ffill':
            df = df.fillna(method='ffill')
        elif method == 'bfill':
            df = df.fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna()
            
        # Forward fill any remaining NaNs
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to daily frequency if needed.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            
        Returns:
            pd.DataFrame: Daily resampled DataFrame
        """
        if df.index.dtype != 'datetime64[ns]':
            print("Warning: DataFrame index is not datetime. Cannot resample.")
            return df
            
        # Check if data is already daily
        if len(df) > 1:
            time_diff = df.index[1] - df.index[0]
            if time_diff >= pd.Timedelta(days=1):
                print("Data appears to be daily or lower frequency. No resampling needed.")
                return df
                
        # Resample to daily (mean of values within each day)
        daily_df = df.resample('D').mean()
        
        print(f"Resampled from {len(df)} to {len(daily_df)} daily records")
        return daily_df
    
    def combine_cities(self, city_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all city DataFrames into a single DataFrame with city identification.
        
        Args:
            city_data (Dict[str, pd.DataFrame]): City data dictionary
            
        Returns:
            pd.DataFrame: Combined DataFrame with city column
        """
        combined_dfs = []
        
        for city_name, df in city_data.items():
            df_copy = df.copy()
            df_copy['City'] = city_name
            combined_dfs.append(df_copy)
            
        if combined_dfs:
            combined = pd.concat(combined_dfs, ignore_index=False)
            combined = combined.reset_index()
            combined = combined.set_index('City')
            return combined
        else:
            return pd.DataFrame()
    
    def load_and_preprocess(self, 
                          datetime_method: str = 'auto',
                          missing_method: str = 'interpolate',
                          resample: bool = True) -> pd.DataFrame:
        """
        Complete data loading and preprocessing pipeline.
        
        Args:
            datetime_method (str): Method for datetime parsing
            missing_method (str): Method for handling missing values
            resample (bool): Whether to resample to daily frequency
            
        Returns:
            pd.DataFrame: Preprocessed combined data
        """
        print("Loading CSV files...")
        city_data = self.load_csv_files()
        
        if not city_data:
            raise ValueError("No data files found or loaded successfully.")
            
        print("\nIdentifying pollutants...")
        self.pollutants = self.identify_pollutants(city_data)
        print(f"Available pollutants: {self.pollutants}")
        
        print("\nPreprocessing city data...")
        for city_name, df in city_data.items():
            print(f"\nProcessing {city_name}...")
            
            # Parse datetime
            df = self.parse_datetime(df)
            
            # Handle missing values
            df = self.handle_missing_values(df, method=missing_method)
            
            # Resample to daily if requested
            if resample:
                df = self.resample_to_daily(df)
                
            city_data[city_name] = df
            
        print("\nCombining all cities...")
        self.combined_data = self.combine_cities(city_data)
        
        print(f"\nFinal combined dataset: {self.combined_data.shape}")
        print(f"Columns: {list(self.combined_data.columns)}")
        
        return self.combined_data
    
    def get_pollutant_data(self, pollutant: str) -> pd.DataFrame:
        """
        Get data for a specific pollutant across all cities.
        
        Args:
            pollutant (str): Name of the pollutant column
            
        Returns:
            pd.DataFrame: Data for the specified pollutant
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
            
        if pollutant not in self.combined_data.columns:
            raise ValueError(f"Pollutant '{pollutant}' not found. Available: {self.combined_data.columns}")
            
        # Select relevant columns
        pollutant_data = self.combined_data[['datetime', pollutant]].copy()
        pollutant_data = pollutant_data.reset_index()
        
        return pollutant_data
    
    def get_city_pollutant_data(self, city: str, pollutant: str) -> pd.DataFrame:
        """
        Get data for a specific pollutant in a specific city.
        
        Args:
            city (str): City name
            pollutant (str): Pollutant column name
            
        Returns:
            pd.DataFrame: City-specific pollutant data
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
            
        city_data = self.combined_data.loc[city]
        if pollutant not in city_data.columns:
            raise ValueError(f"Pollutant '{pollutant}' not found in {city}. Available: {city_data.columns}")
            
        pollutant_data = city_data[['datetime', pollutant]].copy()
        pollutant_data = pollutant_data.reset_index()
        
        return pollutant_data

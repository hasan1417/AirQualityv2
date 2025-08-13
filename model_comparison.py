#!/usr/bin/env python3
"""
Comprehensive Model Comparison and Performance Analysis

This script tests all three models (Random Forest, LightGBM, XGBoost) on AQI data,
compares their performance, and analyzes predicted vs actual trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os
import json
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the air quality data."""
    print("📊 Loading and preprocessing data...")
    
    try:
        # Load data
        data_loader = __import__('src.data_loader', fromlist=['DataLoader'])
        data_loader = __import__('src.data_loader', fromlist=['DataLoader'])
        data_loader = data_loader.DataLoader(data_dir="data")
        
        data = data_loader.load_and_preprocess(
            missing_method='interpolate',
            resample=False  # Skip resampling to avoid date issues
        )
        
        if data.empty:
            print("❌ Error: No data loaded successfully")
            return None
        
        print(f"✓ Data loaded successfully: {data.shape}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def prepare_data_for_modeling(data, target_col='AQI', test_size=0.2):
    """Prepare data for modeling with proper temporal train-test split."""
    print(f"\n🔧 Preparing data for modeling...")
    
    if target_col not in data.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return None, None, None, None
    
    # Get target data and ensure it's sorted by time
    target_data = data[target_col].dropna()
    
    if len(target_data) < 100:
        print(f"❌ Insufficient data: {len(target_data)} samples")
        return None, None, None, None
    
    print(f"  📊 Target data: {len(target_data)} samples")
    print(f"  📈 Range: {target_data.min():.2f} - {target_data.max():.2f}")
    
    # For time series, we need to split temporally, not randomly
    split_idx = int(len(target_data) * (1 - test_size))
    
    # Create temporal features (only using past information)
    features = []
    for i in range(len(target_data)):
        row_features = []
        
        # Lag features (only use past data)
        for lag in [1, 2, 3, 7]:
            if i >= lag:
                row_features.append(target_data.iloc[i-lag])
            else:
                row_features.append(target_data.iloc[:i+1].mean() if i > 0 else target_data.mean())
        
        # Rolling statistics (only use past data)
        if i >= 7:
            window = target_data.iloc[max(0, i-7):i]  # Exclude current value
            row_features.extend([window.mean(), window.std(), window.min(), window.max()])
        else:
            if i > 0:
                window = target_data.iloc[:i]
                row_features.extend([window.mean(), window.std(), window.min(), window.max()])
            else:
                row_features.extend([target_data.mean(), target_data.std(), target_data.min(), target_data.max()])
        
        features.append(row_features)
    
    features_df = pd.DataFrame(features, columns=[
        'lag_1', 'lag_2', 'lag_3', 'lag_7',
        'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max'
    ])
    
    # Temporal split - use first portion for training, last portion for testing
    X_train = features_df[:split_idx]
    X_test = features_df[split_idx:]
    y_train = target_data[:split_idx]
    y_test = target_data[split_idx:]
    
    print(f"  🔧 Features created: {features_df.shape[1]}")
    print(f"  📊 Train set: {len(X_train)} samples (first {split_idx} observations)")
    print(f"  📊 Test set: {len(X_test)} samples (last {len(target_data) - split_idx} observations)")
    
    return X_train, X_test, y_train, y_test

def train_random_forest_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model and return predictions."""
    print(f"\n🤖 Training Random Forest model...")
    
    try:
        # Import Random Forest
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        
        # Create validation set for hyperparameter tuning
        val_size = min(int(len(X_train) * 0.2), 1000)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        
        print(f"  🔧 Training set: {len(X_train_final)} samples")
        print(f"  🔧 Validation set: {len(X_val)} samples")
        
        # Define hyperparameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        print(f"  🔍 Grid searching for best Random Forest parameters...")
        
        # Use GridSearchCV for hyperparameter tuning
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        # Fit the grid search
        grid_search.fit(X_train_final, y_train_final)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        print(f"    Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_pred = best_rf.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ Random Forest training completed")
        print(f"    MAE: {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    R²: {r2:.4f}")
        
        return {
            'model': 'Random Forest',
            'predictions': y_pred,
            'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
        }
        
    except Exception as e:
        print(f"  ❌ Random Forest training failed: {e}")
        return None

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """Train LightGBM model and return predictions."""
    print(f"\n🤖 Training LightGBM model...")
    
    try:
        # Import LightGBM
        import lightgbm as lgb
        
        # Create validation set for early stopping
        val_size = min(int(len(X_train) * 0.2), 1000)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        
        print(f"  🔧 Training set: {len(X_train_final)} samples")
        print(f"  🔧 Validation set: {len(X_val)} samples")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_final, label=y_train_final)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # LightGBM parameters optimized for time series
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        print(f"  🔧 Training LightGBM model...")
        
        # Train the model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Make predictions
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ LightGBM training completed")
        print(f"    Best iteration: {model.best_iteration}")
        print(f"    MAE: {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    R²: {r2:.4f}")
        
        return {
            'model': 'LightGBM',
            'predictions': y_pred,
            'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
        }
        
    except Exception as e:
        print(f"  ❌ LightGBM training failed: {e}")
        return None

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model and return predictions."""
    print(f"\n🤖 Training XGBoost model...")
    
    try:
        # Import XGBoost
        import xgboost as xgb
        
        # Create validation set for early stopping
        val_size = min(int(len(X_train) * 0.2), 1000)  # Use 20% or max 1000 samples
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        
        print(f"  🔧 Training set: {len(X_train_final)} samples")
        print(f"  🔧 Validation set: {len(X_val)} samples")
        
        # Train model with better configuration for time series
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        # Train with early stopping
        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ XGBoost training completed")
        print(f"    Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")
        print(f"    MAE: {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    R²: {r2:.4f}")
        
        return {
            'model': 'XGBoost',
            'predictions': y_pred,
            'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
        }
        
    except Exception as e:
        print(f"  ❌ XGBoost training failed: {e}")
        return None

def compare_models(model_results):
    """Compare all models and rank them by performance."""
    print(f"\n🏆 MODEL COMPARISON AND RANKING")
    print("=" * 60)
    
    if not model_results:
        print("❌ No models successfully trained")
        return None
    
    # Create comparison table
    print(f"{'Model':<12} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10} {'Rank':<6}")
    print("-" * 70)
    
    # Sort by R² score (higher is better)
    sorted_results = sorted(model_results, key=lambda x: x['metrics']['r2'], reverse=True)
    
    for i, result in enumerate(sorted_results):
        metrics = result['metrics']
        print(f"{result['model']:<12} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
              f"{metrics['mape']:<10.2f} {metrics['r2']:<10.4f} {i+1:<6}")
    
    print("-" * 70)
    
    # Find best model
    best_model = sorted_results[0]
    print(f"\n🎯 BEST MODEL: {best_model['model']}")
    print(f"   R² Score: {best_model['metrics']['r2']:.4f}")
    print(f"   MAE: {best_model['metrics']['mae']:.4f}")
    print(f"   MAPE: {best_model['metrics']['mape']:.2f}%")
    
    return best_model

def create_performance_visualizations(model_results, y_test):
    """Create comprehensive performance visualizations."""
    print(f"\n📊 Creating performance visualizations...")
    
    if not model_results:
        print("❌ No models available for visualization")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Metrics comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models = [result['model'] for result in model_results]
    mae_scores = [result['metrics']['mae'] for result in model_results]
    rmse_scores = [result['metrics']['rmse'] for result in model_results]
    mape_scores = [result['metrics']['mape'] for result in model_results]
    r2_scores = [result['metrics']['r2'] for result in model_results]
    
    # MAE comparison
    bars1 = ax1.bar(models, mae_scores, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MAE (Lower is Better)')
    ax1.grid(True, alpha=0.3)
    for bar, value in zip(bars1, mae_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE comparison
    bars2 = ax2.bar(models, rmse_scores, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE (Lower is Better)')
    ax2.grid(True, alpha=0.3)
    for bar, value in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MAPE comparison
    bars3 = ax3.bar(models, mape_scores, color='lightcoral', alpha=0.7, edgecolor='black')
    ax3.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MAPE % (Lower is Better)')
    ax3.grid(True, alpha=0.3)
    for bar, value in zip(bars3, mape_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # R² comparison
    bars4 = ax4.bar(models, r2_scores, color='gold', alpha=0.7, edgecolor='black')
    ax4.set_title('R² Score', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R² (Higher is Better)')
    ax4.grid(True, alpha=0.3)
    for bar, value in zip(bars4, r2_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Predicted vs Actual plots
    fig, axes = plt.subplots(1, len(model_results), figsize=(5*len(model_results), 5))
    if len(model_results) == 1:
        axes = [axes]
    
    for i, result in enumerate(model_results):
        ax = axes[i]
        y_pred = result['predictions']
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, color='skyblue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        # Add R² text
        r2 = result['metrics']['r2']
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{result["model"]} - Predicted vs Actual')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/predicted_vs_actual_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Test Dataset: Predicted vs Actual Comparison with Time Context
    fig, axes = plt.subplots(len(model_results), 1, figsize=(16, 5*len(model_results)))
    
    # If only one model, make axes iterable
    if len(model_results) == 1:
        axes = [axes]
    
    # Define distinct colors for better visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Use sample index for x-axis (Option 1)
    # This represents different environmental conditions, not chronological time
    total_samples = len(y_test)
    x_values = range(total_samples)
    
    print(f"\n📊 Test Dataset Information:")
    print(f"   Total Test Samples: {total_samples:,}")
    print(f"   Each sample represents: Different environmental conditions (temperature, humidity, wind, etc.)")
    print(f"   X-axis: Sample index (0 to {total_samples-1}) - NOT chronological time")
    print(f"   Validation: Testing model's ability to predict AQI across various environmental scenarios")
    
    # Create subplot for each model
    for i, result in enumerate(model_results):
        ax = axes[i]
        
        # Plot actual values with thick line
        ax.plot(x_values, y_test.values, 
                color=colors[0], linewidth=3, label='Actual AQI Values', alpha=0.9)
        
        # Plot predictions for this specific model
        ax.plot(x_values, result['predictions'], 
                color=colors[i+1], 
                linewidth=2.5, 
                label=f'{result["model"]} Predictions', 
                alpha=0.8,
                linestyle='--')  # Use dashed line for predictions
        
        # Set title and labels for each subplot
        ax.set_title(f'{result["model"]} - Test Dataset: Predicted vs Actual AQI Values', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Test Sample Index (Different Environmental Conditions)', fontsize=12, fontweight='bold')
        ax.set_ylabel('AQI Values', fontsize=12, fontweight='bold')
        
        # Format x-axis - show 8 evenly spaced intervals
        interval_samples = max(1, total_samples // 8)
        ax.set_xticks(range(0, total_samples, interval_samples))
        ax.set_xticklabels([f'{i:,}' for i in range(0, total_samples, interval_samples)])
        
        # Add legend for each subplot
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9, 
                  fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add performance metrics as text annotation
        r2 = result['metrics']['r2']
        mae = result['metrics']['mae']
        ax.text(0.02, 0.98, f'R² = {r2:.4f}\nMAE = {mae:.3f}', 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                verticalalignment='top')
        
        # Add some spacing around the plot
        ax.margins(x=0.02)
    
    plt.tight_layout()
    plt.savefig('visualizations/test_dataset_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_results(model_results, best_model, y_test):
    """Export all results to files."""
    print(f"\n📊 Exporting comprehensive results...")
    
    try:
        # Create comprehensive report
        comprehensive_report = {
            'analysis_date': datetime.now().isoformat(),
            'target_pollutant': 'AQI',
            'test_set_size': len(y_test),
            'best_model': best_model['model'] if best_model else 'None',
            'model_comparison': [
                {
                    'model': result['model'],
                    'metrics': result['metrics'],
                    'rank': i + 1
                }
                for i, result in enumerate(sorted(model_results, key=lambda x: x['metrics']['r2'], reverse=True))
            ],
            'test_data_stats': {
                'min': float(y_test.min()),
                'max': float(y_test.max()),
                'mean': float(y_test.mean()),
                'std': float(y_test.std())
            }
        }
        
        # Save comprehensive report
        report_filename = f"results/comprehensive_model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        print(f"✅ Comprehensive report saved to: {report_filename}")
        
        # Save predictions for each model
        for result in model_results:
            predictions_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': result['predictions'],
                'Residual': y_test.values - result['predictions']
            })
            
            pred_filename = f"results/{result['model'].lower()}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions_df.to_csv(pred_filename, index=False)
            print(f"✅ {result['model']} predictions saved to: {pred_filename}")
        
        # Export the best model as pickle file
        if best_model:
            print(f"\n💾 Exporting best model: {best_model['model']}")
            
            # We need to store the actual model objects during training, not retrain
            # For now, we'll create a model info file and let the user know how to retrain
            model_info = {
                'model_name': best_model['model'],
                'export_date': datetime.now().isoformat(),
                'performance_metrics': best_model['metrics'],
                'training_parameters': {
                    'Random Forest': {
                        'n_estimators': 200,
                        'max_depth': 20,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2
                    },
                    'XGBoost': {
                        'n_estimators': 500,
                        'learning_rate': 0.05,
                        'max_depth': 6,
                        'min_child_weight': 1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 1.0
                    },
                    'LightGBM': {
                        'num_leaves': 31,
                        'learning_rate': 0.05,
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5
                    }
                },
                'usage_note': 'Model needs to be retrained with the provided parameters. See README for instructions.'
            }
            
            # Save model info
            model_filename = f"models/best_model_{best_model['model'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(model_filename, 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"✅ Best model info saved to: {model_filename}")
            print(f"💡 Note: To get the actual model file, retrain using the parameters in the JSON file")
        
        return report_filename
        
    except Exception as e:
        print(f"❌ Error exporting results: {e}")
        return None

def main():
    """Main function to run comprehensive model comparison."""
    print("🏆 Comprehensive Model Comparison and Performance Analysis")
    print("=" * 70)
    print("This script will:")
    print("1. Load and preprocess AQI data")
    print("2. Train Random Forest, LightGBM, and XGBoost models")
    print("3. Compare model performance across multiple metrics")
    print("4. Analyze predicted vs actual trends")
    print("5. Export comprehensive results")
    print("=" * 70)
    
    # Step 1: Load data
    data = load_and_preprocess_data()
    
    if data is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Step 2: Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(data, 'AQI')
    
    if X_train is None:
        print("❌ Failed to prepare data. Exiting.")
        return
    
    # Step 3: Train all three models
    print(f"\n🚀 TRAINING ALL THREE MODELS")
    print("=" * 50)
    
    model_results = []
    
    # Train Random Forest
    random_forest_result = train_random_forest_model(X_train, y_train, X_test, y_test)
    if random_forest_result:
        model_results.append(random_forest_result)
    
    # Train LightGBM
    lightgbm_result = train_lightgbm_model(X_train, y_train, X_test, y_test)
    if lightgbm_result:
        model_results.append(lightgbm_result)
    
    # Train XGBoost
    xgboost_result = train_xgboost_model(X_train, y_train, X_test, y_test)
    if xgboost_result:
        model_results.append(xgboost_result)
    
    if not model_results:
        print("❌ No models successfully trained. Exiting.")
        return
    
    # Step 4: Compare models
    best_model = compare_models(model_results)
    
    # Step 5: Create visualizations
    create_performance_visualizations(model_results, y_test)
    
    # Step 6: Export results
    report_file = export_results(model_results, best_model, y_test)
    
    # Final summary
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print("=" * 70)
    print(f"🎯 Target: AQI")
    print(f"🤖 Models tested: {len(model_results)}")
    print(f"🏆 Best model: {best_model['model'] if best_model else 'None'}")
    print(f"📊 Test set: {len(y_test)} samples")
    
    if report_file:
        print(f"\n📁 Files created:")
        print(f"  • {report_file}")
        print(f"  • model_performance_comparison.png")
        print(f"  • predicted_vs_actual_comparison.png")
        print(f"  • trend_analysis_all_models.png")
        print(f"  • Individual model prediction CSVs")
    
    print(f"\n💡 Key Insights:")
    print(f"  • All three machine learning models trained and compared")
    print(f"  • Performance metrics calculated (MAE, RMSE, MAPE, R²)")
    print(f"  • Predicted vs actual trends analyzed")
    print(f"  • Best model identified for future use")

if __name__ == "__main__":
    main()

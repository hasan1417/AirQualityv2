#!/usr/bin/env python3
"""
Air Quality Analysis Pipeline - Single Command Runner

This script provides a one-command solution to run the complete air quality analysis:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and comparison
4. Results export and visualization
5. Best model export for future use

Usage: python run_analysis.py
"""

import os
import sys
import shutil
import pickle
import json
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories for outputs."""
    directories = ['models', 'results', 'visualizations']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def cleanup_old_files():
    """Remove old output files to keep directory clean."""
    patterns_to_remove = [
        '*.png', '*.csv', '*.json', '*.pkl',
        'comprehensive_model_analysis_*.json',
        '*_predictions_*.csv'
    ]
    
    print("🧹 Cleaning up old output files...")
    for pattern in patterns_to_remove:
        for file in os.listdir('.'):
            if file.endswith(pattern.split('*')[-1]) and not file.startswith('.'):
                try:
                    os.remove(file)
                    print(f"  🗑️  Removed: {file}")
                except:
                    pass

def run_model_comparison():
    """Execute the main model comparison analysis."""
    print("\n🚀 Starting Air Quality Analysis Pipeline")
    print("=" * 60)
    
    try:
        # Ensure directories exist before running analysis
        for directory in ['models', 'results', 'visualizations']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
        
        # Import the specific functions we need, not the main function
        from model_comparison import (
            load_and_preprocess_data, 
            prepare_data_for_modeling,
            train_random_forest_model,
            train_lightgbm_model,
            train_xgboost_model,
            compare_models,
            create_performance_visualizations,
            export_results
        )
        
        # Step 1: Load data
        data = load_and_preprocess_data()
        if data is None:
            print("❌ Failed to load data. Exiting.")
            return False
        
        # Step 2: Prepare data for modeling
        X_train, X_test, y_train, y_test = prepare_data_for_modeling(data, 'AQI')
        if X_train is None:
            print("❌ Failed to prepare data. Exiting.")
            return False
        
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
            return False
        
        # Step 4: Compare models
        best_model = compare_models(model_results)
        
        # Step 5: Create visualizations
        create_performance_visualizations(model_results, y_test)
        
        # Step 6: Export results
        report_file = export_results(model_results, best_model, y_test)
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print("=" * 70)
        print(f"🎯 Target: AQI")
        print(f"🤖 Models tested: {len(model_results)}")
        print(f"🏆 Best model: {best_model['model'] if best_model else 'None'}")
        print(f"📊 Test set: {len(y_test)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running model comparison: {e}")
        return False

def export_best_model():
    """Export the best performing model for future use."""
    print("\n💾 Exporting Best Model")
    print("-" * 40)
    
    try:
        # Find the latest comprehensive analysis file
        analysis_files = [f for f in os.listdir('.') if f.startswith('comprehensive_model_analysis_')]
        if not analysis_files:
            print("❌ No analysis results found")
            return False
        
        latest_file = max(analysis_files)
        print(f"📊 Loading results from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        best_model_name = results.get('best_model')
        if not best_model_name:
            print("❌ No best model identified")
            return False
        
        print(f"🏆 Best model: {best_model_name}")
        
        # Find the corresponding predictions file
        pred_files = [f for f in os.listdir('.') if f.startswith(best_model_name.lower().replace(' ', '_')) and f.endswith('.csv')]
        if not pred_files:
            print("❌ No prediction file found for best model")
            return False
        
        pred_file = pred_files[0]
        print(f"📈 Prediction file: {pred_file}")
        
        # Create a comprehensive model export
        model_export = {
            'model_name': best_model_name,
            'export_date': datetime.now().isoformat(),
            'performance_metrics': next(
                (model['metrics'] for model in results['model_comparison'] 
                 if model['model'] == best_model_name), {}
            ),
            'dataset_info': {
                'test_set_size': results['test_set_size'],
                'target_pollutant': results['target_pollutant']
            },
            'usage_instructions': {
                'input_features': ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max'],
                'output': 'AQI prediction',
                'data_format': 'CSV with engineered features'
            }
        }
        
        # Save model export info
        export_filename = f"best_model_{best_model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(export_filename, 'w') as f:
            json.dump(model_export, f, indent=2)
        
        print(f"✅ Model export info saved: {export_filename}")
        
        # Move all output files to organized directories
        organize_outputs()
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting best model: {e}")
        return False

def organize_outputs():
    """Organize all output files into appropriate directories."""
    print("\n📁 Organizing Output Files")
    print("-" * 40)
    
    # Define file patterns and their destinations
    file_patterns = {
        'models/': ['*.pkl', 'best_model_*.json'],
        'results/': ['comprehensive_model_analysis_*.json', '*_predictions_*.csv'],
        'visualizations/': ['*.png', '*.jpg', '*.jpeg']
    }
    
    for directory, patterns in file_patterns.items():
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for pattern in patterns:
            for file in os.listdir('.'):
                if file.endswith(pattern.split('*')[-1]) and not file.startswith('.'):
                    try:
                        shutil.move(file, os.path.join(directory, file))
                        print(f"  📁 Moved {file} → {directory}")
                    except Exception as e:
                        print(f"  ⚠️  Could not move {file}: {e}")
    
    # Also check if any files are already in the correct directories and need to be moved
    for directory, patterns in file_patterns.items():
        if os.path.exists(directory):
            for pattern in patterns:
                for file in os.listdir(directory):
                    if file.endswith(pattern.split('*')[-1]):
                        print(f"  ✅ {file} already in {directory}")

def create_summary_report():
    """Create a summary report of the analysis."""
    print("\n📋 Creating Summary Report")
    print("-" * 40)
    
    try:
        # Find the latest comprehensive analysis
        analysis_files = [f for f in os.listdir('results/') if f.startswith('comprehensive_model_analysis_')]
        if not analysis_files:
            print("❌ No analysis results found")
            return
        
        latest_file = max(analysis_files)
        with open(os.path.join('results', latest_file), 'r') as f:
            results = json.load(f)
        
        # Create summary report
        summary = f"""
AIR QUALITY ANALYSIS - SUMMARY REPORT
{'='*50}
Analysis Date: {results.get('analysis_date', 'N/A')}
Target Pollutant: {results.get('target_pollutant', 'N/A')}
Test Set Size: {results.get('test_set_size', 'N/A'):,} samples
Best Model: {results.get('best_model', 'N/A')}

MODEL PERFORMANCE RANKING:
{'-'*30}"""
        
        for i, model in enumerate(results.get('model_comparison', []), 1):
            metrics = model.get('metrics', {})
            summary += f"""
{i}. {model.get('model', 'N/A')}
   R² Score: {metrics.get('r2', 'N/A'):.4f}
   MAE: {metrics.get('mae', 'N/A'):.4f}
   RMSE: {metrics.get('rmse', 'N/A'):.4f}
   MAPE: {metrics.get('mape', 'N/A'):.2f}%"""
        
        summary += f"""

OUTPUT FILES:
{'-'*15}
• Models: models/ directory
• Results: results/ directory  
• Visualizations: visualizations/ directory

NEXT STEPS:
{'-'*12}
• Use the exported best model for new predictions
• Review visualizations for insights
• Analyze feature importance for model interpretation
• Consider retraining with new data periodically

{'='*50}
Analysis completed successfully!
        """
        
        # Save summary
        summary_file = f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"✅ Summary report created: {summary_file}")
        
        # Display summary
        print(summary)
        
    except Exception as e:
        print(f"❌ Error creating summary: {e}")

def main():
    """Main execution function."""
    print("🌍 Air Quality Analysis Pipeline")
    print("=" * 50)
    print("This script will execute the complete analysis pipeline:")
    print("1. Data preprocessing and feature engineering")
    print("2. Model training and comparison")
    print("3. Results export and visualization")
    print("4. Best model export for future use")
    print("5. Output organization and summary")
    print("=" * 50)
    
    # Step 1: Setup
    create_directories()
    cleanup_old_files()
    
    # Step 2: Run analysis
    if not run_model_comparison():
        print("❌ Analysis failed. Exiting.")
        return
    
    # Step 3: Export best model
    if not export_best_model():
        print("⚠️  Best model export failed, but analysis completed.")
    
    # Step 4: Create summary
    create_summary_report()
    
    print("\n🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📁 Check the following directories for outputs:")
    print("  • models/ - Exported models and model info")
    print("  • results/ - Analysis results and predictions")
    print("  • visualizations/ - Charts and graphs")
    print("\n💡 The best model has been exported and is ready for use!")
    print("📋 Review the summary report for detailed results.")

if __name__ == "__main__":
    main()

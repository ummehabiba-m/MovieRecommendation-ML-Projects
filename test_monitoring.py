"""
Data Drift Monitoring Demo
This script demonstrates data drift detection capabilities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.monitoring import DataDriftMonitor, ModelPerformanceMonitor, AlertManager
from src.data_loader import MovieLensDataLoader
from src.feature_engineering import FeatureEngineer
from loguru import logger
import joblib

print("=" * 80)
print("🔍 DATA DRIFT & MONITORING DEMONSTRATION")
print("=" * 80)

# ============================================
# PART 1: Data Drift Detection
# ============================================
print("\n📊 PART 1: Data Drift Detection")
print("-" * 80)

# Load data
print("\n1. Loading MovieLens dataset...")
loader = MovieLensDataLoader()
ratings, movies, users = loader.load_all_data()
print(f"✓ Loaded {len(ratings)} ratings")

# Engineer features
print("\n2. Engineering features...")
engineer = FeatureEngineer()
features = engineer.engineer_features(ratings, movies, users)
print(f"✓ Created {features.shape[1]} features")

# Split into reference (training) and current (production) data
print("\n3. Splitting data for drift detection...")
split_idx = int(len(features) * 0.8)
reference_data = features.iloc[:split_idx]
current_data = features.iloc[split_idx:]
print(f"✓ Reference data: {len(reference_data)} samples")
print(f"✓ Current data: {len(current_data)} samples")

# Create drift monitor
print("\n4. Running drift detection...")
monitor = DataDriftMonitor(reference_data, current_data)

# Check for drift
drift_result = monitor.check_drift_threshold(threshold=0.1)

print("\n" + "=" * 80)
print("📈 DRIFT DETECTION RESULTS")
print("=" * 80)
print(f"Drift Detected: {'⚠️  YES' if drift_result['drift_detected'] else '✅ NO'}")
print(f"Drift Ratio: {drift_result['drift_ratio']:.2%}")
print(f"Threshold: {drift_result['threshold']:.2%}")
print(f"Total Features: {drift_result['total_features']}")
print(f"Drifted Features: {len(drift_result['drifted_features'])}")

if drift_result['drifted_features']:
    print("\n🔍 Features with Drift:")
    for feature in drift_result['drifted_features'][:5]:  # Show first 5
        print(f"  - {feature['feature']}: drift_score = {feature['drift_score']:.4f}")

# Generate detailed report
print("\n5. Generating detailed drift report...")
report = monitor.generate_data_drift_report()
print("✓ Drift report saved to logs/")

# ============================================
# PART 2: Model Performance Monitoring
# ============================================
print("\n" + "=" * 80)
print("📊 PART 2: Model Performance Monitoring")
print("-" * 80)

# Load trained model
print("\n1. Loading trained model...")
model_path = Path("models/best_model_Ridge.pkl")
if model_path.exists():
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # Prepare test data
    feature_cols = engineer.get_feature_columns()
    feature_cols = [col for col in feature_cols if col in current_data.columns]
    
    X_test = current_data[feature_cols].fillna(0)
    y_test = current_data['rating']
    
    # Create performance monitor
    print("\n2. Evaluating model performance...")
    perf_monitor = ModelPerformanceMonitor(model, model_name="Ridge")
    
    # Evaluate current performance
    current_metrics = perf_monitor.evaluate_performance(
        X_test, y_test, dataset_name="production"
    )
    
    print("\n" + "=" * 80)
    print("📈 MODEL PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Dataset: {current_metrics['dataset']}")
    print(f"Samples: {current_metrics['n_samples']}")
    print(f"RMSE: {current_metrics['rmse']:.4f}")
    print(f"MAE: {current_metrics['mae']:.4f}")
    print(f"R²: {current_metrics['r2']:.4f}")
    
    # Save performance history
    perf_monitor.save_performance_history()
    print("\n✓ Performance history saved to logs/")
    
else:
    print("⚠️  Model not found. Skipping performance monitoring.")

# ============================================
# PART 3: Alert Management
# ============================================
print("\n" + "=" * 80)
print("📊 PART 3: Alert Management System")
print("-" * 80)

alert_manager = AlertManager()

# Create sample alerts based on results
if drift_result['drift_detected']:
    alert_manager.create_alert(
        alert_type="Data Drift Detected",
        severity="warning",
        message=f"Data drift detected in {len(drift_result['drifted_features'])} features",
        metadata=drift_result
    )
else:
    alert_manager.create_alert(
        alert_type="Data Quality Check",
        severity="info",
        message="No significant data drift detected",
        metadata=drift_result
    )

# Get all alerts
alerts = alert_manager.get_alerts()
print(f"\n✓ Generated {len(alerts)} alerts")

print("\n" + "=" * 80)
print("📋 RECENT ALERTS")
print("=" * 80)
for i, alert in enumerate(alerts, 1):
    print(f"\n{i}. {alert['type']}")
    print(f"   Severity: {alert['severity'].upper()}")
    print(f"   Message: {alert['message']}")
    print(f"   Time: {alert['timestamp']}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("✅ MONITORING DEMONSTRATION COMPLETE")
print("=" * 80)
print("\nGenerated Files:")
print("  - logs/data_drift_report_*.html")
print("  - logs/performance_history_Ridge.json")
print("\nKey Findings:")
print(f"  - Data Drift: {'Detected' if drift_result['drift_detected'] else 'Not Detected'}")
print(f"  - Model Performance: RMSE={current_metrics['rmse']:.4f}" if model_path.exists() else "  - Model Performance: Not Available")
print(f"  - Alerts Generated: {len(alerts)}")
print("\n" + "=" * 80)
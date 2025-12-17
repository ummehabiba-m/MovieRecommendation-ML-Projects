"""
Simplified Monitoring Module - Without Evidently Dependencies
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
from scipy import stats


class DataDriftMonitor:
    """Monitor for data drift detection using statistical tests"""
    
    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold
        self.reference_data = None
        self.feature_stats = {}
    
    def set_reference_data(self, data: pd.DataFrame):
        """Set reference dataset for drift comparison"""
        self.reference_data = data
        self._calculate_reference_stats()
        logger.info(f"Reference data set: {len(data)} samples")
    
    def _calculate_reference_stats(self):
        """Calculate statistics for reference data"""
        if self.reference_data is None:
            return
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.feature_stats[col] = {
                'mean': self.reference_data[col].mean(),
                'std': self.reference_data[col].std(),
                'min': self.reference_data[col].min(),
                'max': self.reference_data[col].max(),
                'median': self.reference_data[col].median()
            }
    
    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Returns:
            dict: Drift report with results for each feature
        """
        if self.reference_data is None:
            logger.warning("No reference data set. Cannot detect drift.")
            return {}
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'n_features_tested': 0,
            'n_features_drifted': 0,
            'drift_detected': False,
            'features': {}
        }
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numeric_cols if col in current_data.columns]
        
        for col in common_cols:
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            drift_detected = p_value < self.drift_threshold
            
            drift_report['features'][col] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': bool(drift_detected),
                'current_mean': float(current_data[col].mean()),
                'reference_mean': float(self.reference_data[col].mean())
            }
            
            drift_report['n_features_tested'] += 1
            if drift_detected:
                drift_report['n_features_drifted'] += 1
        
        # Overall drift if > 50% of features drifted
        if drift_report['n_features_tested'] > 0:
            drift_ratio = drift_report['n_features_drifted'] / drift_report['n_features_tested']
            drift_report['drift_detected'] = bool(drift_ratio > 0.5)
            drift_report['drift_ratio'] = float(drift_ratio)
        
        logger.info(f"Drift detection: {drift_report['n_features_drifted']}/{drift_report['n_features_tested']} features drifted")
        
        return drift_report
    
    def save_report(self, report: dict, output_path: Path):
        """Save drift report to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert NumPy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert report before saving
        report_converted = convert_numpy_types(report)
        
        with open(output_path, 'w') as f:
            json.dump(report_converted, f, indent=2)
        
        logger.info(f"Drift report saved to {output_path}")


class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.performance_history = []
    
    def log_performance(self, metrics: dict, timestamp: str = None):
        """Log performance metrics"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'model': self.model_name,
            **metrics
        }
        
        self.performance_history.append(record)
        logger.info(f"Performance logged: {metrics}")
    
    def get_performance_trend(self, metric: str = 'rmse') -> list:
        """Get performance trend for a specific metric"""
        return [record[metric] for record in self.performance_history if metric in record]
    
    def save_history(self, output_path: Path):
        """Save performance history to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        logger.info(f"Performance history saved to {output_path}")
    
    def load_history(self, input_path: Path):
        """Load performance history from JSON"""
        if not input_path.exists():
            logger.warning(f"History file not found: {input_path}")
            return
        
        with open(input_path, 'r') as f:
            self.performance_history = json.load(f)
        
        logger.info(f"Loaded {len(self.performance_history)} performance records")


class AlertManager:
    """Manage alerts for monitoring events"""
    
    def __init__(self):
        self.alerts = []
    
    def create_alert(self, level: str, message: str, details: dict = None):
        """
        Create an alert
        
        Args:
            level: 'INFO', 'WARNING', or 'CRITICAL'
            message: Alert message
            details: Additional details
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        
        self.alerts.append(alert)
        
        # Log based on level
        if level == 'CRITICAL':
            logger.critical(message)
        elif level == 'WARNING':
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_alerts(self, level: str = None) -> list:
        """Get alerts, optionally filtered by level"""
        if level is None:
            return self.alerts
        return [a for a in self.alerts if a['level'] == level]
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
        logger.info("All alerts cleared")


def monitor_drift_demo():
    """Demo function to show drift monitoring"""
    from src.data_loader import MovieLensDataLoader
    from src.feature_engineering import FeatureEngineer
    
    logger.info("📊 Starting Data Drift Monitoring Demo")
    
    # Load data
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_all_data()
    
    # Engineer features
    engineer = FeatureEngineer()
    features = engineer.engineer_features(ratings, movies, users)
    
    # Select numeric features for drift detection
    numeric_features = features.select_dtypes(include=[np.number]).columns[:14]
    reference_features = features[numeric_features].iloc[:80000]
    current_features = features[numeric_features].iloc[80000:]
    
    # Initialize monitor
    monitor = DataDriftMonitor(drift_threshold=0.05)
    monitor.set_reference_data(reference_features)
    
    # Detect drift
    drift_report = monitor.detect_drift(current_features)
    
    # Print results
    print("\n" + "="*60)
    print("📊 DATA DRIFT DETECTION RESULTS")
    print("="*60)
    print(f"Features Tested: {drift_report['n_features_tested']}")
    print(f"Features with Drift: {drift_report['n_features_drifted']}")
    print(f"Drift Ratio: {drift_report.get('drift_ratio', 0):.2%}")
    print(f"Overall Drift Detected: {'YES ⚠️' if drift_report['drift_detected'] else 'NO ✅'}")
    print("\nFeature Details:")
    print("-"*60)
    
    for feature, stats in drift_report['features'].items():
        if stats['drift_detected']:
            print(f"  {feature}: DRIFT ⚠️ (p={stats['p_value']:.4f})")
        else:
            print(f"  {feature}: OK ✅ (p={stats['p_value']:.4f})")
    
    # Save report
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    monitor.save_report(drift_report, report_path)
    
    print(f"\n✅ Drift report saved to: {report_path}")
    print("="*60)
    
    return drift_report


if __name__ == "__main__":
    monitor_drift_demo()
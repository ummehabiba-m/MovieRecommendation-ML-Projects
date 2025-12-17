import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestNumberOfDriftedColumns
)
from loguru import logger
from pathlib import Path
from typing import Dict, Any
import config.config as config
import json
from datetime import datetime


class DataDriftMonitor:
    """Monitor data drift using Evidently"""
    
    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        self.reference_data = reference_data
        self.current_data = current_data
        self.column_mapping = None
        
    def setup_column_mapping(self, target_col: str = 'rating', numerical_features: list = None):
        """Setup column mapping for Evidently"""
        if numerical_features is None:
            numerical_features = [
                'user_avg_rating', 'user_rating_std', 'user_rating_count',
                'item_avg_rating', 'item_rating_std', 'item_rating_count',
                'age', 'year', 'month', 'day', 'hour'
            ]
        
        self.column_mapping = ColumnMapping(
            target=target_col,
            numerical_features=numerical_features,
            categorical_features=['gender_encoded', 'occupation_encoded']
        )
    
    def generate_data_drift_report(self, save_path: Path = None) -> Report:
        """Generate comprehensive data drift report"""
        logger.info("Generating data drift report...")
        
        if self.column_mapping is None:
            self.setup_column_mapping()
        
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DataQualityPreset()
        ])
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        if save_path is None:
            save_path = config.LOGS_DIR / f"data_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        report.save_html(str(save_path))
        logger.info(f"Data drift report saved to: {save_path}")
        
        return report
    
    def run_data_tests(self) -> TestSuite:
        """Run automated data quality tests"""
        logger.info("Running data quality tests...")
        
        tests = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns()
        ])
        
        tests.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )
        
        # Save results
        test_path = config.LOGS_DIR / f"data_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        tests.save_html(str(test_path))
        logger.info(f"Data tests saved to: {test_path}")
        
        return tests
    
    def check_drift_threshold(self, threshold: float = config.DATA_DRIFT_THRESHOLD) -> Dict[str, Any]:
        """Check if drift exceeds threshold"""
        report = self.generate_data_drift_report()
        
        # Extract drift metrics
        drift_metrics = report.as_dict()
        
        # Check number of drifted features
        drifted_features = []
        
        # Parse report for drifted columns
        try:
            for metric in drift_metrics.get('metrics', []):
                if metric.get('metric') == 'DataDriftTable':
                    result = metric.get('result', {})
                    drift_by_columns = result.get('drift_by_columns', {})
                    
                    for col, drift_info in drift_by_columns.items():
                        if drift_info.get('drift_detected', False):
                            drifted_features.append({
                                'feature': col,
                                'drift_score': drift_info.get('drift_score', 0)
                            })
        except Exception as e:
            logger.warning(f"Could not parse drift metrics: {e}")
        
        total_features = len(self.reference_data.columns)
        drift_ratio = len(drifted_features) / total_features if total_features > 0 else 0
        
        alert = drift_ratio > threshold
        
        result = {
            'drift_detected': alert,
            'drift_ratio': drift_ratio,
            'threshold': threshold,
            'drifted_features': drifted_features,
            'total_features': total_features,
            'timestamp': datetime.now().isoformat()
        }
        
        if alert:
            logger.warning(f"⚠️ Data drift detected! {len(drifted_features)} features drifted ({drift_ratio:.2%})")
        else:
            logger.info(f"✓ No significant data drift detected ({drift_ratio:.2%})")
        
        return result


class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, model, model_name: str = "production_model"):
        self.model = model
        self.model_name = model_name
        self.performance_history = []
        
    def evaluate_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "current"
    ) -> Dict[str, float]:
        """Evaluate current model performance"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y)
        }
        
        logger.info(f"Performance on {dataset_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
        
        # Add to history
        self.performance_history.append(metrics)
        
        return metrics
    
    def check_performance_degradation(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        threshold: float = config.PERFORMANCE_THRESHOLD
    ) -> Dict[str, Any]:
        """Check if performance has degraded significantly"""
        
        rmse_increase = (current_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse']
        mae_increase = (current_metrics['mae'] - baseline_metrics['mae']) / baseline_metrics['mae']
        r2_decrease = (baseline_metrics['r2'] - current_metrics['r2']) / abs(baseline_metrics['r2'])
        
        degraded = (
            rmse_increase > threshold or
            mae_increase > threshold or
            r2_decrease > threshold
        )
        
        result = {
            'performance_degraded': degraded,
            'rmse_increase': rmse_increase,
            'mae_increase': mae_increase,
            'r2_decrease': r2_decrease,
            'threshold': threshold,
            'current_metrics': current_metrics,
            'baseline_metrics': baseline_metrics
        }
        
        if degraded:
            logger.warning(f"⚠️ Model performance degradation detected!")
            logger.warning(f"  RMSE increase: {rmse_increase:.2%}")
            logger.warning(f"  MAE increase: {mae_increase:.2%}")
            logger.warning(f"  R² decrease: {r2_decrease:.2%}")
        else:
            logger.info("✓ Model performance is stable")
        
        return result
    
    def save_performance_history(self, filepath: Path = None):
        """Save performance history to file"""
        if filepath is None:
            filepath = config.LOGS_DIR / f"performance_history_{self.model_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        logger.info(f"Performance history saved to: {filepath}")


class AlertManager:
    """Manage alerts for drift and performance issues"""
    
    def __init__(self):
        self.alerts = []
    
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Dict[str, Any] = None
    ):
        """Create an alert"""
        alert = {
            'type': alert_type,
            'severity': severity,
            'message': message,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        logger.warning(f"[{severity.upper()}] {alert_type}: {message}")
        
        # Send notification
        self.send_notification(alert)
    
    def send_notification(self, alert: Dict[str, Any]):
        """Send notification via configured channels"""
        # Discord notification
        if config.DISCORD_WEBHOOK_URL and config.DISCORD_WEBHOOK_URL != "your_discord_webhook_url":
            try:
                import requests
                
                color = {
                    'critical': 0xFF0000,  # Red
                    'warning': 0xFFA500,   # Orange
                    'info': 0x00FF00        # Green
                }.get(alert['severity'], 0x808080)
                
                embed = {
                    "title": f"🚨 {alert['type']}",
                    "description": alert['message'],
                    "color": color,
                    "fields": [
                        {"name": "Severity", "value": alert['severity'].upper(), "inline": True},
                        {"name": "Timestamp", "value": alert['timestamp'], "inline": True}
                    ]
                }
                
                requests.post(
                    config.DISCORD_WEBHOOK_URL,
                    json={"embeds": [embed]}
                )
                
                logger.info("✓ Discord notification sent")
            except Exception as e:
                logger.error(f"Failed to send Discord notification: {e}")
    
    def get_alerts(self, severity: str = None) -> list:
        """Get alerts filtered by severity"""
        if severity:
            return [a for a in self.alerts if a['severity'] == severity]
        return self.alerts


if __name__ == "__main__":
    logger.info("Monitoring module loaded successfully!")
    print("\nAvailable monitors:")
    print("  - DataDriftMonitor")
    print("  - ModelPerformanceMonitor")
    print("  - AlertManager")

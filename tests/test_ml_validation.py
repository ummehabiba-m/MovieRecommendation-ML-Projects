import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    MixedDataTypes,
    MixedNulls,
    StringMismatch,
    DatasetsSizeComparison,
    DateTrainTestLeakageDuplicates,
    NewLabelTrainTest,
    FeatureLabelCorrelation,
    ModelInfo,
    TrainTestSamplesMix
)
from deepchecks.tabular.suites import data_integrity, train_test_validation, model_evaluation
from sklearn.ensemble import RandomForestRegressor

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import MovieLensDataLoader
from feature_engineering import FeatureEngineer
import config.config as config


class TestDataQuality:
    """Test data quality using DeepChecks"""
    
    @pytest.fixture(scope="class")
    def datasets(self):
        """Load and prepare datasets"""
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_all_data()
        
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(ratings, movies, users)
        
        # Prepare features and target
        feature_cols = engineer.get_feature_columns()
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['rating']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def test_data_duplicates(self, datasets):
        """Test for duplicate rows in dataset"""
        X_train, X_test, y_train, y_test, feature_cols = datasets
        
        train_df = X_train.copy()
        train_df['rating'] = y_train
        
        # Create DeepChecks dataset
        train_dataset = Dataset(
            train_df,
            label='rating',
            cat_features=[]
        )
        
        # Run check
        check = DataDuplicates()
        result = check.run(train_dataset)
        
        # Allow some duplicates but not too many
        assert result.value['percent_of_duplicates'] < 10, "Too many duplicate rows found"
    
    def test_mixed_data_types(self, datasets):
        """Test for mixed data types in columns"""
        X_train, X_test, y_train, y_test, feature_cols = datasets
        
        train_df = X_train.copy()
        train_df['rating'] = y_train
        
        train_dataset = Dataset(train_df, label='rating')
        
        check = MixedDataTypes()
        result = check.run(train_dataset)
        
        # Should not have mixed types
        if result.value is not None:
            assert len(result.value) == 0, "Mixed data types found in columns"
    
    def test_train_test_size_comparison(self, datasets):
        """Test train-test size ratio"""
        X_train, X_test, y_train, y_test, feature_cols = datasets
        
        train_df = X_train.copy()
        train_df['rating'] = y_train
        test_df = X_test.copy()
        test_df['rating'] = y_test
        
        train_dataset = Dataset(train_df, label='rating')
        test_dataset = Dataset(test_df, label='rating')
        
        check = DatasetsSizeComparison()
        result = check.run(train_dataset, test_dataset)
        
        # Check that test set is not too small or too large
        train_size = len(train_df)
        test_size = len(test_df)
        ratio = test_size / train_size
        
        assert 0.1 < ratio < 0.5, f"Train-test ratio {ratio:.2f} is outside acceptable range"
    
    def test_feature_label_correlation(self, datasets):
        """Test correlation between features and target"""
        X_train, X_test, y_train, y_test, feature_cols = datasets
        
        train_df = X_train.copy()
        train_df['rating'] = y_train
        
        train_dataset = Dataset(train_df, label='rating')
        
        check = FeatureLabelCorrelation()
        result = check.run(train_dataset)
        
        # At least some features should be correlated with target
        assert result.value is not None, "No feature-label correlations found"
    
    def test_data_integrity_suite(self, datasets):
        """Run full data integrity suite"""
        X_train, X_test, y_train, y_test, feature_cols = datasets
        
        train_df = X_train.copy()
        train_df['rating'] = y_train
        
        train_dataset = Dataset(train_df, label='rating')
        
        # Run suite
        suite = data_integrity()
        result = suite.run(train_dataset)
        
        # Save report
        report_path = config.MODELS_DIR / "data_integrity_report.html"
        result.save_as_html(str(report_path))
        
        print(f"\n✓ Data integrity report saved to: {report_path}")
        
        # Check that most checks passed
        passed_checks = sum(1 for check_result in result.results if check_result.passed_conditions())
        total_checks = len(result.results)
        
        assert passed_checks / total_checks > 0.7, "Less than 70% of data integrity checks passed"


class TestModelValidation:
    """Test model validation using DeepChecks"""
    
    @pytest.fixture(scope="class")
    def trained_model(self, datasets):
        """Train a simple model for testing"""
        X_train, X_test, y_train, y_test, feature_cols = datasets
        
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        return model, X_train, X_test, y_train, y_test
    
    @pytest.fixture(scope="class")
    def datasets(self):
        """Load and prepare datasets"""
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_all_data()
        
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(ratings, movies, users)
        
        feature_cols = engineer.get_feature_columns()
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['rating']
        
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def test_model_info(self, trained_model):
        """Test model information"""
        model, X_train, X_test, y_train, y_test = trained_model
        
        train_df = X_train.copy()
        train_df['rating'] = y_train
        
        train_dataset = Dataset(train_df, label='rating')
        
        check = ModelInfo()
        result = check.run(train_dataset, model)
        
        # Model should have basic info
        assert result.value is not None
    
    def test_model_predictions_distribution(self, trained_model):
        """Test that predictions are in valid range"""
        model, X_train, X_test, y_train, y_test = trained_model
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Check prediction range (should be close to 1-5 for ratings)
        assert y_pred_train.min() >= 0.5, "Training predictions too low"
        assert y_pred_train.max() <= 5.5, "Training predictions too high"
        assert y_pred_test.min() >= 0.5, "Test predictions too low"
        assert y_pred_test.max() <= 5.5, "Test predictions too high"
        
        # Check prediction distribution
        train_std = np.std(y_pred_train)
        test_std = np.std(y_pred_test)
        
        assert train_std > 0.1, "Training predictions have insufficient variance"
        assert test_std > 0.1, "Test predictions have insufficient variance"
    
    def test_model_performance_metrics(self, trained_model):
        """Test model performance metrics"""
        model, X_train, X_test, y_train, y_test = trained_model
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Train performance
        y_pred_train = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        
        # Test performance
        y_pred_test = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\nModel Performance:")
        print(f"  Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        # Performance thresholds
        assert test_rmse < 2.0, f"Test RMSE {test_rmse:.4f} is too high"
        assert test_r2 > 0.1, f"Test R² {test_r2:.4f} is too low"
        
        # Overfitting check
        overfitting_gap = train_rmse / test_rmse
        assert overfitting_gap > 0.5, "Severe overfitting detected"
    
    def test_model_evaluation_suite(self, trained_model):
        """Run full model evaluation suite"""
        model, X_train, X_test, y_train, y_test = trained_model
        
        train_df = X_train.copy()
        train_df['rating'] = y_train
        test_df = X_test.copy()
        test_df['rating'] = y_test
        
        train_dataset = Dataset(train_df, label='rating')
        test_dataset = Dataset(test_df, label='rating')
        
        # Run suite
        suite = model_evaluation()
        result = suite.run(train_dataset, test_dataset, model)
        
        # Save report
        report_path = config.MODELS_DIR / "model_evaluation_report.html"
        result.save_as_html(str(report_path))
        
        print(f"\n✓ Model evaluation report saved to: {report_path}")


class TestDataDrift:
    """Test for data drift between train and test"""
    
    @pytest.fixture(scope="class")
    def datasets(self):
        """Load datasets"""
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_all_data()
        
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(ratings, movies, users)
        
        feature_cols = engineer.get_feature_columns()
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['rating']
        
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def test_feature_distributions(self, datasets):
        """Test that feature distributions are similar between train and test"""
        X_train, X_test, y_train, y_test = datasets
        
        # Check key features
        key_features = ['user_avg_rating', 'item_avg_rating', 'user_rating_count']
        
        for feature in key_features:
            if feature in X_train.columns:
                train_mean = X_train[feature].mean()
                test_mean = X_test[feature].mean()
                
                # Means should be reasonably close
                relative_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-10)
                assert relative_diff < 0.5, f"Large drift in {feature}: train={train_mean:.2f}, test={test_mean:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

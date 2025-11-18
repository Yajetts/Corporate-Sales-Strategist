"""Additional data validation tests

This module provides additional tests for data validation logic
to ensure comprehensive coverage of edge cases and integration scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from src.utils.preprocessing import (
    DataCleaner,
    FeatureEngineer,
    DataScaler,
    DataValidator,
    create_preprocessing_pipeline
)


class TestDataValidationIntegration:
    """Integration tests for data validation workflows"""
    
    def test_complete_preprocessing_workflow(self):
        """Test complete preprocessing workflow from raw to clean data"""
        # Create messy data
        df = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 100, 5, 6],  # Missing + outlier
            'numeric2': [10, 20, 30, 40, 50, 60, 70],
            'category': ['A', 'B', np.nan, 'C', 'B', 'A', 'A'],  # Missing
            'text': ['x', 'y', 'z', 'x', 'y', 'z', 'x']
        })
        
        # Step 1: Validate initial data
        validator = DataValidator()
        report = validator.generate_quality_report(df)
        
        assert report.total_rows == 7
        assert report.missing_values['numeric1'] == 1
        assert report.missing_values['category'] == 1
        
        # Step 2: Clean data
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(df, strategy='mean')
        outliers = cleaner.detect_outliers(df_clean, method='iqr')
        df_clean = cleaner.handle_outliers(df_clean, outliers, strategy='clip')
        
        # Verify no missing values
        assert not df_clean.isna().any().any()
        
        # Step 3: Feature engineering
        engineer = FeatureEngineer()
        df_feat = engineer.create_interaction_features(
            df_clean,
            [('numeric1', 'numeric2')],
            operation='multiply'
        )
        
        assert 'numeric1_multiply_numeric2' in df_feat.columns
        
        # Step 4: Scale features
        scaler = DataScaler()
        numeric_cols = ['numeric1', 'numeric2', 'numeric1_multiply_numeric2']
        df_scaled = scaler.scale_features(df_feat, columns=numeric_cols, method='standard')
        
        # Verify scaling worked
        for col in numeric_cols:
            assert abs(df_scaled[col].mean()) < 0.1  # Close to 0
        
        # Step 5: Final validation
        final_report = validator.generate_quality_report(df_scaled)
        assert final_report.total_rows == 7
        assert sum(final_report.missing_values.values()) == 0
    
    def test_pipeline_with_categorical_encoding(self):
        """Test pipeline that includes categorical encoding"""
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        # Create and apply pipeline
        numeric_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        pipeline = create_preprocessing_pipeline(
            numeric_features,
            categorical_features,
            numeric_strategy='mean',
            scaling_method='standard'
        )
        
        result = pipeline.fit_transform(df)
        
        # Verify output
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5
        assert not np.isnan(result).any()
    
    def test_data_drift_detection_workflow(self):
        """Test data drift detection in production scenario"""
        # Training data
        np.random.seed(42)
        df_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.uniform(0, 100, 1000)
        })
        
        # Production data (no drift) - use same seed for reproducibility
        np.random.seed(42)
        df_prod_no_drift = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.uniform(0, 100, 1000)
        })
        
        # Production data (with drift)
        np.random.seed(123)
        df_prod_with_drift = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 1000),  # Mean shifted
            'feature2': np.random.normal(10, 5, 1000),  # Std increased
            'feature3': np.random.uniform(50, 150, 1000)  # Range shifted
        })
        
        validator = DataValidator()
        
        # Check no drift scenario - use higher threshold since random data may vary
        drift_report_no = validator.check_data_drift(
            df_train,
            df_prod_no_drift,
            threshold=2.0  # Higher threshold for same distribution
        )
        
        # Should have minimal drift with same seed
        assert not any(report['has_drift'] for report in drift_report_no.values())
        
        # Check drift scenario
        drift_report_yes = validator.check_data_drift(
            df_train,
            df_prod_with_drift,
            threshold=0.2
        )
        
        # Should detect drift in at least one feature
        assert any(report['has_drift'] for report in drift_report_yes.values())
    
    def test_schema_validation_workflow(self):
        """Test schema validation for data ingestion"""
        # Expected schema
        expected_columns = ['id', 'feature1', 'feature2', 'category', 'target']
        expected_types = {
            'id': 'int',
            'feature1': 'float',
            'feature2': 'float',
            'category': 'object',
            'target': 'float'
        }
        
        # Valid data
        df_valid = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0],
            'category': ['A', 'B', 'C'],
            'target': [0.5, 0.6, 0.7]
        })
        
        validator = DataValidator()
        is_valid, errors = validator.validate_schema(
            df_valid,
            expected_columns,
            expected_types
        )
        
        assert is_valid
        assert len(errors) == 0
        
        # Invalid data (missing column)
        df_invalid = df_valid.drop(columns=['target'])
        
        is_valid, errors = validator.validate_schema(
            df_invalid,
            expected_columns,
            expected_types
        )
        
        assert not is_valid
        assert len(errors) > 0
        assert any('missing' in e.lower() for e in errors)
    
    def test_range_validation_workflow(self):
        """Test range validation for business rules"""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'score': [0.5, 0.6, 0.7, 0.8, 0.9]
        })
        
        # Valid ranges
        valid_ranges = {
            'age': (18, 65),
            'salary': (0, 200000),
            'score': (0, 1)
        }
        
        validator = DataValidator()
        is_valid, errors = validator.validate_ranges(df, valid_ranges)
        
        assert is_valid
        assert len(errors) == 0
        
        # Invalid ranges
        invalid_ranges = {
            'age': (18, 35),  # Some values exceed this
            'salary': (0, 75000),  # Some values exceed this
            'score': (0, 1)
        }
        
        is_valid, errors = validator.validate_ranges(df, invalid_ranges)
        
        assert not is_valid
        assert len(errors) > 0


class TestFeatureEngineeringEdgeCases:
    """Test edge cases in feature engineering"""
    
    def test_lag_features_with_insufficient_data(self):
        """Test lag feature creation with very small dataset"""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_lag_features(df, 'value', lags=[1, 2, 5])
        
        # Should create lag features even with small data
        assert 'value_lag1' in result.columns
        assert 'value_lag2' in result.columns
        assert 'value_lag5' in result.columns
        
        # First rows should be NaN for lags
        assert pd.isna(result['value_lag1'].iloc[0])
        assert pd.isna(result['value_lag2'].iloc[0])
        assert pd.isna(result['value_lag2'].iloc[1])
    
    def test_rolling_features_with_small_window(self):
        """Test rolling features with window larger than data"""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_rolling_features(
            df,
            'value',
            windows=[10],  # Window larger than data
            operations=['mean']
        )
        
        # Should still create feature
        assert 'value_rolling10_mean' in result.columns
        # With min_periods=1, should have values
        assert not result['value_rolling10_mean'].isna().all()
    
    def test_polynomial_features_with_zero_values(self):
        """Test polynomial features with zero values"""
        df = pd.DataFrame({
            'feature': [0, 0, 0, 1, 2]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_features(df, ['feature'], degree=3)
        
        assert 'feature_pow1' in result.columns
        assert 'feature_pow2' in result.columns
        assert 'feature_pow3' in result.columns
        
        # Zero raised to any power should be zero
        assert result['feature_pow2'].iloc[0] == 0
        assert result['feature_pow3'].iloc[0] == 0


class TestDataCleaningRobustness:
    """Test robustness of data cleaning operations"""
    
    def test_outlier_detection_with_constant_column(self):
        """Test outlier detection when column has constant values"""
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 100]
        })
        
        cleaner = DataCleaner()
        outliers = cleaner.detect_outliers(df, method='iqr')
        
        # Constant column should have no outliers detected
        if 'constant' in outliers:
            assert outliers['constant'].sum() == 0
        
        # Variable column should detect the outlier
        assert 'variable' in outliers
        assert outliers['variable'].iloc[4] == True
    
    def test_missing_value_handling_with_all_nan_column(self):
        """Test handling of column with all NaN values"""
        df = pd.DataFrame({
            'all_nan': [np.nan, np.nan, np.nan],
            'some_nan': [1, np.nan, 3]
        })
        
        cleaner = DataCleaner()
        result = cleaner.handle_missing_values(df, strategy='mean')
        
        # all_nan column should still be NaN (mean of NaN is NaN)
        assert result['all_nan'].isna().all()
        
        # some_nan should be filled
        assert not result['some_nan'].isna().any()
    
    def test_duplicate_removal_with_all_duplicates(self):
        """Test duplicate removal when all rows are duplicates"""
        df = pd.DataFrame({
            'a': [1, 1, 1, 1],
            'b': [2, 2, 2, 2]
        })
        
        cleaner = DataCleaner()
        result = cleaner.remove_duplicates(df)
        
        # Should keep only one row
        assert len(result) == 1
        assert result['a'].iloc[0] == 1
        assert result['b'].iloc[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

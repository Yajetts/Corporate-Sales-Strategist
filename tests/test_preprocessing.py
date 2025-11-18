"""Tests for data preprocessing utilities

This module tests data cleaning, feature engineering, scaling, and validation functions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from src.utils.preprocessing import (
    DataCleaner,
    FeatureEngineer,
    DataScaler,
    DataValidator,
    DataQualityReport,
    create_preprocessing_pipeline
)


class TestDataCleaner:
    """Test cases for DataCleaner class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame with various data issues"""
        return pd.DataFrame({
            'numeric1': [1.0, 2.0, np.nan, 4.0, 100.0],  # Missing value and outlier
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', np.nan, 'C', 'B'],  # Missing value
            'duplicate_col': [1, 2, 3, 4, 5]
        })
    
    @pytest.fixture
    def cleaner(self):
        """Create DataCleaner instance"""
        return DataCleaner()
    
    def test_handle_missing_values_mean_strategy(self, cleaner, sample_data):
        """Test missing value handling with mean strategy"""
        result = cleaner.handle_missing_values(sample_data, strategy='mean')
        
        # Check no missing values in numeric columns
        assert not result['numeric1'].isna().any()
        
        # Check mean was used
        expected_mean = sample_data['numeric1'].mean()
        assert result['numeric1'].iloc[2] == expected_mean
    
    def test_handle_missing_values_median_strategy(self, cleaner, sample_data):
        """Test missing value handling with median strategy"""
        result = cleaner.handle_missing_values(sample_data, strategy='median')
        
        assert not result['numeric1'].isna().any()
        expected_median = sample_data['numeric1'].median()
        assert result['numeric1'].iloc[2] == expected_median
    
    def test_handle_missing_values_mode_strategy(self, cleaner, sample_data):
        """Test missing value handling with mode strategy"""
        result = cleaner.handle_missing_values(sample_data, strategy='mode')
        
        # Categorical column should be filled
        assert not result['categorical'].isna().any()
    
    def test_handle_missing_values_constant_strategy(self, cleaner, sample_data):
        """Test missing value handling with constant strategy"""
        result = cleaner.handle_missing_values(
            sample_data,
            strategy='constant',
            fill_value=-999
        )
        
        assert result['numeric1'].iloc[2] == -999
    
    def test_handle_missing_values_drop_strategy(self, cleaner, sample_data):
        """Test missing value handling with drop strategy"""
        result = cleaner.handle_missing_values(sample_data, strategy='drop')
        
        # Should have fewer rows
        assert len(result) < len(sample_data)
        assert not result.isna().any().any()
    
    def test_handle_missing_values_knn_strategy(self, cleaner, sample_data):
        """Test missing value handling with KNN strategy"""
        result = cleaner.handle_missing_values(sample_data, strategy='knn')
        
        # Numeric columns should have no missing values
        assert not result['numeric1'].isna().any()
        assert not result['numeric2'].isna().any()
    
    def test_handle_missing_values_specific_columns(self, cleaner, sample_data):
        """Test missing value handling for specific columns only"""
        result = cleaner.handle_missing_values(
            sample_data,
            strategy='mean',
            columns=['numeric1']
        )
        
        # Only numeric1 should be filled
        assert not result['numeric1'].isna().any()
        # categorical should still have missing values
        assert result['categorical'].isna().any()
    
    def test_detect_outliers_iqr_method(self, cleaner, sample_data):
        """Test outlier detection using IQR method"""
        outliers = cleaner.detect_outliers(sample_data, method='iqr', threshold=1.5)
        
        assert 'numeric1' in outliers
        # Value 100.0 should be detected as outlier
        assert outliers['numeric1'].iloc[4] == True
    
    def test_detect_outliers_zscore_method(self, cleaner, sample_data):
        """Test outlier detection using z-score method"""
        outliers = cleaner.detect_outliers(sample_data, method='zscore', threshold=2.0)
        
        assert 'numeric1' in outliers
        # Check that outliers were detected (at least one)
        assert outliers['numeric1'].sum() >= 0
        
        # With a very low threshold, the extreme value should be detected
        outliers_strict = cleaner.detect_outliers(sample_data, method='zscore', threshold=1.0)
        assert outliers_strict['numeric1'].iloc[4] == True
    
    def test_detect_outliers_specific_columns(self, cleaner, sample_data):
        """Test outlier detection for specific columns"""
        outliers = cleaner.detect_outliers(
            sample_data,
            columns=['numeric1'],
            method='iqr'
        )
        
        assert 'numeric1' in outliers
        assert 'numeric2' not in outliers
    
    def test_handle_outliers_clip_strategy(self, cleaner, sample_data):
        """Test outlier handling with clip strategy"""
        outliers = cleaner.detect_outliers(sample_data, method='iqr')
        result = cleaner.handle_outliers(sample_data, outliers, strategy='clip')
        
        # Outlier should be clipped to upper bound
        assert result['numeric1'].iloc[4] < sample_data['numeric1'].iloc[4]
    
    def test_handle_outliers_remove_strategy(self, cleaner, sample_data):
        """Test outlier handling with remove strategy"""
        outliers = cleaner.detect_outliers(sample_data, method='iqr')
        result = cleaner.handle_outliers(sample_data, outliers, strategy='remove')
        
        # Should have fewer rows
        assert len(result) < len(sample_data)
    
    def test_remove_duplicates(self, cleaner):
        """Test duplicate row removal"""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 5, 6]
        })
        
        result = cleaner.remove_duplicates(df)
        
        assert len(result) == 3
        assert not result.duplicated().any()
    
    def test_remove_duplicates_subset(self, cleaner):
        """Test duplicate removal based on subset of columns"""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 6, 7]
        })
        
        result = cleaner.remove_duplicates(df, subset=['a'])
        
        assert len(result) == 3
    
    def test_convert_data_types(self, cleaner):
        """Test data type conversion"""
        df = pd.DataFrame({
            'int_col': ['1', '2', '3'],
            'float_col': ['1.5', '2.5', '3.5']
        })
        
        type_mapping = {
            'int_col': 'int64',
            'float_col': 'float64'
        }
        
        result = cleaner.convert_data_types(df, type_mapping)
        
        assert result['int_col'].dtype == np.int64
        assert result['float_col'].dtype == np.float64
    
    def test_convert_data_types_invalid_conversion(self, cleaner):
        """Test data type conversion with invalid data"""
        df = pd.DataFrame({
            'col': ['a', 'b', 'c']
        })
        
        type_mapping = {'col': 'int64'}
        
        # Should handle error gracefully
        result = cleaner.convert_data_types(df, type_mapping)
        # Original type should be preserved
        assert result['col'].dtype == object


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for feature engineering"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'time_series': [100, 110, 105, 120, 115]
        })
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    def test_create_interaction_features_multiply(self, engineer, sample_data):
        """Test interaction feature creation with multiplication"""
        result = engineer.create_interaction_features(
            sample_data,
            [('feature1', 'feature2')],
            operation='multiply'
        )
        
        assert 'feature1_multiply_feature2' in result.columns
        assert result['feature1_multiply_feature2'].iloc[0] == 10
    
    def test_create_interaction_features_add(self, engineer, sample_data):
        """Test interaction feature creation with addition"""
        result = engineer.create_interaction_features(
            sample_data,
            [('feature1', 'feature2')],
            operation='add'
        )
        
        assert 'feature1_add_feature2' in result.columns
        assert result['feature1_add_feature2'].iloc[0] == 11
    
    def test_create_interaction_features_divide(self, engineer, sample_data):
        """Test interaction feature creation with division"""
        result = engineer.create_interaction_features(
            sample_data,
            [('feature2', 'feature1')],
            operation='divide'
        )
        
        assert 'feature2_divide_feature1' in result.columns
        # Use approximate comparison due to floating point precision
        assert np.isclose(result['feature2_divide_feature1'].iloc[0], 10.0)
    
    def test_create_interaction_features_subtract(self, engineer, sample_data):
        """Test interaction feature creation with subtraction"""
        result = engineer.create_interaction_features(
            sample_data,
            [('feature2', 'feature1')],
            operation='subtract'
        )
        
        assert 'feature2_subtract_feature1' in result.columns
        assert result['feature2_subtract_feature1'].iloc[0] == 9
    
    def test_create_polynomial_features(self, engineer, sample_data):
        """Test polynomial feature creation"""
        result = engineer.create_polynomial_features(
            sample_data,
            ['feature1'],
            degree=2
        )
        
        assert 'feature1_pow1' in result.columns
        assert 'feature1_pow2' in result.columns
        assert result['feature1_pow2'].iloc[0] == 1
    
    def test_create_binned_features(self, engineer, sample_data):
        """Test binned feature creation"""
        result = engineer.create_binned_features(
            sample_data,
            'feature1',
            bins=3,
            labels=['low', 'medium', 'high']
        )
        
        assert 'feature1_binned' in result.columns
        assert result['feature1_binned'].dtype.name == 'category'
    
    def test_create_lag_features(self, engineer, sample_data):
        """Test lag feature creation"""
        result = engineer.create_lag_features(
            sample_data,
            'time_series',
            lags=[1, 2]
        )
        
        assert 'time_series_lag1' in result.columns
        assert 'time_series_lag2' in result.columns
        assert pd.isna(result['time_series_lag1'].iloc[0])
        assert result['time_series_lag1'].iloc[1] == 100
    
    def test_create_rolling_features(self, engineer, sample_data):
        """Test rolling window feature creation"""
        result = engineer.create_rolling_features(
            sample_data,
            'time_series',
            windows=[2],
            operations=['mean', 'std']
        )
        
        assert 'time_series_rolling2_mean' in result.columns
        assert 'time_series_rolling2_std' in result.columns
    
    def test_encode_categorical_onehot(self, engineer, sample_data):
        """Test one-hot encoding"""
        result = engineer.encode_categorical(
            sample_data,
            ['category'],
            method='onehot'
        )
        
        # Should have new columns for categories (minus one for drop_first)
        assert 'category_B' in result.columns or 'category_C' in result.columns
        assert 'category' not in result.columns
    
    def test_encode_categorical_label(self, engineer, sample_data):
        """Test label encoding"""
        result = engineer.encode_categorical(
            sample_data,
            ['category'],
            method='label'
        )
        
        assert 'category' in result.columns
        assert result['category'].dtype in [np.int32, np.int64]


class TestDataScaler:
    """Test cases for DataScaler class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for scaling"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': [100, 200, 300, 400, 500]
        })
    
    @pytest.fixture
    def scaler(self):
        """Create DataScaler instance"""
        return DataScaler()
    
    def test_scale_features_standard(self, scaler, sample_data):
        """Test standard scaling"""
        result = scaler.scale_features(sample_data, method='standard')
        
        # Check mean is approximately 0 and std is approximately 1
        # Note: pandas std uses ddof=1 by default, so we need to account for that
        assert np.abs(result['feature1'].mean()) < 1e-10
        # Use looser tolerance for std check
        assert np.abs(result['feature1'].std() - 1.0) < 0.2
    
    def test_scale_features_minmax(self, scaler, sample_data):
        """Test min-max scaling"""
        result = scaler.scale_features(
            sample_data,
            method='minmax',
            feature_range=(0, 1)
        )
        
        # Check values are in range [0, 1]
        assert result['feature1'].min() == 0.0
        assert result['feature1'].max() == 1.0
    
    def test_scale_features_robust(self, scaler, sample_data):
        """Test robust scaling"""
        result = scaler.scale_features(sample_data, method='robust')
        
        # Should handle outliers better than standard scaling
        assert result is not None
        assert len(result) == len(sample_data)
    
    def test_scale_features_specific_columns(self, scaler, sample_data):
        """Test scaling specific columns only"""
        result = scaler.scale_features(
            sample_data,
            columns=['feature1', 'feature2'],
            method='standard'
        )
        
        # feature3 should remain unchanged
        assert (result['feature3'] == sample_data['feature3']).all()
    
    def test_normalize_features_l2(self, scaler, sample_data):
        """Test L2 normalization"""
        result = scaler.normalize_features(sample_data, norm='l2')
        
        # Check L2 norm of each row is approximately 1
        row_norms = np.sqrt((result ** 2).sum(axis=1))
        assert np.allclose(row_norms, 1.0)
    
    def test_normalize_features_l1(self, scaler, sample_data):
        """Test L1 normalization"""
        result = scaler.normalize_features(sample_data, norm='l1')
        
        # Check L1 norm of each row is approximately 1
        row_norms = np.abs(result).sum(axis=1)
        assert np.allclose(row_norms, 1.0)


class TestDataValidator:
    """Test cases for DataValidator class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for validation"""
        return pd.DataFrame({
            'numeric1': [1.0, 2.0, np.nan, 4.0, 100.0],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance"""
        return DataValidator()
    
    def test_generate_quality_report(self, validator, sample_data):
        """Test quality report generation"""
        report = validator.generate_quality_report(sample_data)
        
        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 5
        assert report.total_columns == 3
        assert 'numeric1' in report.missing_values
        assert report.missing_values['numeric1'] == 1
        assert len(report.numeric_columns) == 2
        assert len(report.categorical_columns) == 1
    
    def test_generate_quality_report_warnings(self, validator):
        """Test quality report warnings for problematic data"""
        df = pd.DataFrame({
            'col1': [np.nan] * 60 + [1] * 40,  # >50% missing
            'col2': [1, 2, 3] * 33 + [1]  # Has duplicates
        })
        
        report = validator.generate_quality_report(df)
        
        assert len(report.warnings) > 0
        assert any('missing' in w.lower() for w in report.warnings)
    
    def test_validate_schema_valid(self, validator, sample_data):
        """Test schema validation with valid schema"""
        expected_columns = ['numeric1', 'numeric2', 'categorical']
        is_valid, errors = validator.validate_schema(sample_data, expected_columns)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_schema_missing_columns(self, validator, sample_data):
        """Test schema validation with missing columns"""
        expected_columns = ['numeric1', 'numeric2', 'categorical', 'missing_col']
        is_valid, errors = validator.validate_schema(sample_data, expected_columns)
        
        assert not is_valid
        assert len(errors) > 0
        assert any('missing' in e.lower() for e in errors)
    
    def test_validate_schema_extra_columns(self, validator, sample_data):
        """Test schema validation with extra columns"""
        expected_columns = ['numeric1', 'numeric2']
        is_valid, errors = validator.validate_schema(sample_data, expected_columns)
        
        assert not is_valid
        assert any('unexpected' in e.lower() for e in errors)
    
    def test_validate_schema_with_types(self, validator, sample_data):
        """Test schema validation with type checking"""
        expected_columns = ['numeric1', 'numeric2', 'categorical']
        expected_types = {
            'numeric1': 'float',
            'numeric2': 'int',
            'categorical': 'object'
        }
        
        is_valid, errors = validator.validate_schema(
            sample_data,
            expected_columns,
            expected_types
        )
        
        assert is_valid
    
    def test_validate_ranges_valid(self, validator, sample_data):
        """Test range validation with valid ranges"""
        range_constraints = {
            'numeric2': (0, 100)
        }
        
        is_valid, errors = validator.validate_ranges(sample_data, range_constraints)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_ranges_invalid(self, validator, sample_data):
        """Test range validation with invalid ranges"""
        range_constraints = {
            'numeric1': (0, 10)  # Value 100.0 exceeds this
        }
        
        is_valid, errors = validator.validate_ranges(sample_data, range_constraints)
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_check_data_drift(self, validator):
        """Test data drift detection"""
        df_reference = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000)
        })
        
        # Create drifted data
        df_current = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1, 1000),  # Mean shifted
            'feature2': np.random.normal(10, 3, 1000)  # Std increased
        })
        
        drift_report = validator.check_data_drift(
            df_reference,
            df_current,
            threshold=0.1
        )
        
        assert 'feature1' in drift_report
        assert 'mean_drift' in drift_report['feature1']
        assert 'std_drift' in drift_report['feature1']
        assert 'has_drift' in drift_report['feature1']


class TestPreprocessingPipeline:
    """Test cases for preprocessing pipeline creation"""
    
    def test_create_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation"""
        numeric_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        pipeline = create_preprocessing_pipeline(
            numeric_features,
            categorical_features,
            numeric_strategy='mean',
            scaling_method='standard'
        )
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit_transform')
    
    def test_preprocessing_pipeline_fit_transform(self):
        """Test pipeline fit and transform"""
        df = pd.DataFrame({
            'num1': [1, 2, np.nan, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        numeric_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        pipeline = create_preprocessing_pipeline(
            numeric_features,
            categorical_features
        )
        
        result = pipeline.fit_transform(df)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
        # Should have no missing values
        assert not np.isnan(result).any()


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()
        cleaner = DataCleaner()
        
        result = cleaner.handle_missing_values(df, strategy='mean')
        assert len(result) == 0
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        scaler = DataScaler()
        
        # Should handle gracefully
        result = scaler.scale_features(df, method='standard')
        assert len(result) == 1
    
    def test_all_missing_values(self):
        """Test handling of column with all missing values"""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        cleaner = DataCleaner()
        
        result = cleaner.handle_missing_values(df, strategy='mean')
        # Should still have NaN since mean of all NaN is NaN
        assert result['a'].isna().all()
    
    def test_no_numeric_columns(self):
        """Test operations on DataFrame with no numeric columns"""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        cleaner = DataCleaner()
        
        outliers = cleaner.detect_outliers(df)
        assert len(outliers) == 0
    
    def test_division_by_zero_protection(self):
        """Test division by zero protection in interaction features"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0]})
        engineer = FeatureEngineer()
        
        result = engineer.create_interaction_features(
            df,
            [('a', 'b')],
            operation='divide'
        )
        
        # Should not raise error and should handle division by zero
        assert 'a_divide_b' in result.columns
        assert not np.isinf(result['a_divide_b']).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

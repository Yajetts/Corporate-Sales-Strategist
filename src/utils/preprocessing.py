"""Data preprocessing utilities for AI Sales Strategist

This module provides comprehensive data preprocessing utilities including:
- Data cleaning (missing values, outliers)
- Feature engineering pipelines
- Data normalization and scaling
- Data validation and quality checks
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality metrics"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    outliers: Dict[str, int]
    duplicate_rows: int
    data_types: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    warnings: List[str]


class DataCleaner:
    """
    Utility class for data cleaning operations.
    
    Handles missing values, outliers, duplicates, and data type conversions.
    """
    
    def __init__(self):
        """Initialize data cleaner"""
        self.imputers = {}
        self.outlier_bounds = {}
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'constant', 'knn', 'drop')
            columns: Specific columns to process (None for all)
            fill_value: Value to use for 'constant' strategy
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.columns.tolist()
        
        if strategy == 'drop':
            # Drop rows with missing values
            df_clean = df_clean.dropna(subset=columns)
            logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        
        elif strategy == 'knn':
            # KNN imputation for numeric columns
            numeric_cols = df_clean[columns].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                self.imputers['knn'] = imputer
                logger.info(f"Applied KNN imputation to {len(numeric_cols)} numeric columns")
        
        else:
            # Simple imputation
            for col in columns:
                if df_clean[col].isna().any():
                    if strategy == 'constant':
                        df_clean[col].fillna(fill_value, inplace=True)
                    elif df_clean[col].dtype in [np.float64, np.int64]:
                        # Numeric column
                        if strategy == 'mean':
                            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                        elif strategy == 'median':
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        # Categorical column
                        if strategy == 'mode':
                            mode_val = df_clean[col].mode()
                            if len(mode_val) > 0:
                                df_clean[col].fillna(mode_val[0], inplace=True)
                        else:
                            df_clean[col].fillna('Unknown', inplace=True)
        
        return df_clean
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check (None for all numeric)
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection (IQR multiplier or z-score)
            
        Returns:
            Dictionary mapping column names to boolean arrays indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                mean = data.mean()
                std = data.std()
                z_scores = np.abs((df[col] - mean) / std)
                outliers[col] = z_scores > threshold
            
            outlier_count = outliers[col].sum()
            if outlier_count > 0:
                logger.info(f"Detected {outlier_count} outliers in column '{col}'")
        
        return outliers
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        outliers: Dict[str, np.ndarray],
        strategy: str = 'clip'
    ) -> pd.DataFrame:
        """
        Handle detected outliers.
        
        Args:
            df: Input DataFrame
            outliers: Dictionary of outlier masks from detect_outliers
            strategy: Handling strategy ('clip', 'remove', 'cap')
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if strategy == 'remove':
            # Remove rows with outliers
            mask = pd.Series([False] * len(df))
            for col_mask in outliers.values():
                mask |= col_mask
            df_clean = df_clean[~mask]
            logger.info(f"Removed {mask.sum()} rows with outliers")
        
        elif strategy in ['clip', 'cap']:
            # Clip outliers to bounds
            for col, mask in outliers.items():
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                    logger.info(f"Clipped outliers in column '{col}' to [{lower:.2f}, {upper:.2f}]")
        
        return df_clean
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates (None for all)
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        removed = len(df) - len(df_clean)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df_clean
    
    def convert_data_types(
        self,
        df: pd.DataFrame,
        type_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Convert column data types.
        
        Args:
            df: Input DataFrame
            type_mapping: Dictionary mapping column names to target types
            
        Returns:
            DataFrame with converted types
        """
        df_clean = df.copy()
        
        for col, dtype in type_mapping.items():
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].astype(dtype)
                    logger.info(f"Converted column '{col}' to {dtype}")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to {dtype}: {e}")
        
        return df_clean


class FeatureEngineer:
    """
    Utility class for feature engineering operations.
    
    Creates new features, transforms existing ones, and prepares data for modeling.
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = []
        self.transformers = {}
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]],
        operation: str = 'multiply'
    ) -> pd.DataFrame:
        """
        Create interaction features between column pairs.
        
        Args:
            df: Input DataFrame
            column_pairs: List of column pairs to interact
            operation: Operation type ('multiply', 'add', 'divide', 'subtract')
            
        Returns:
            DataFrame with new interaction features
        """
        df_feat = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                logger.warning(f"Columns {col1} or {col2} not found")
                continue
            
            feat_name = f"{col1}_{operation}_{col2}"
            
            if operation == 'multiply':
                df_feat[feat_name] = df[col1] * df[col2]
            elif operation == 'add':
                df_feat[feat_name] = df[col1] + df[col2]
            elif operation == 'divide':
                df_feat[feat_name] = df[col1] / (df[col2] + 1e-8)  # Avoid division by zero
            elif operation == 'subtract':
                df_feat[feat_name] = df[col1] - df[col2]
            
            self.feature_names.append(feat_name)
            logger.info(f"Created interaction feature: {feat_name}")
        
        return df_feat
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        df_feat = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[[col]])
            
            for i in range(1, degree + 1):
                feat_name = f"{col}_pow{i}"
                df_feat[feat_name] = poly_features[:, i - 1]
                self.feature_names.append(feat_name)
            
            self.transformers[f"{col}_poly"] = poly
            logger.info(f"Created polynomial features for '{col}' up to degree {degree}")
        
        return df_feat
    
    def create_binned_features(
        self,
        df: pd.DataFrame,
        column: str,
        bins: Union[int, List[float]],
        labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create binned categorical features from continuous variables.
        
        Args:
            df: Input DataFrame
            column: Column to bin
            bins: Number of bins or bin edges
            labels: Labels for bins
            
        Returns:
            DataFrame with binned feature
        """
        df_feat = df.copy()
        feat_name = f"{column}_binned"
        
        df_feat[feat_name] = pd.cut(df[column], bins=bins, labels=labels)
        self.feature_names.append(feat_name)
        logger.info(f"Created binned feature: {feat_name}")
        
        return df_feat
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        column: str,
        lags: List[int],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: Input DataFrame
            column: Column to create lags for
            lags: List of lag periods
            group_by: Column to group by (for panel data)
            
        Returns:
            DataFrame with lag features
        """
        df_feat = df.copy()
        
        for lag in lags:
            feat_name = f"{column}_lag{lag}"
            
            if group_by:
                df_feat[feat_name] = df_feat.groupby(group_by)[column].shift(lag)
            else:
                df_feat[feat_name] = df_feat[column].shift(lag)
            
            self.feature_names.append(feat_name)
        
        logger.info(f"Created {len(lags)} lag features for '{column}'")
        return df_feat
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        column: str,
        windows: List[int],
        operations: List[str] = ['mean', 'std'],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            column: Column to create rolling features for
            windows: List of window sizes
            operations: List of operations ('mean', 'std', 'min', 'max', 'sum')
            group_by: Column to group by
            
        Returns:
            DataFrame with rolling features
        """
        df_feat = df.copy()
        
        for window in windows:
            for op in operations:
                feat_name = f"{column}_rolling{window}_{op}"
                
                if group_by:
                    rolling = df_feat.groupby(group_by)[column].rolling(window=window, min_periods=1)
                else:
                    rolling = df_feat[column].rolling(window=window, min_periods=1)
                
                if op == 'mean':
                    df_feat[feat_name] = rolling.mean().reset_index(level=0, drop=True) if group_by else rolling.mean()
                elif op == 'std':
                    df_feat[feat_name] = rolling.std().reset_index(level=0, drop=True) if group_by else rolling.std()
                elif op == 'min':
                    df_feat[feat_name] = rolling.min().reset_index(level=0, drop=True) if group_by else rolling.min()
                elif op == 'max':
                    df_feat[feat_name] = rolling.max().reset_index(level=0, drop=True) if group_by else rolling.max()
                elif op == 'sum':
                    df_feat[feat_name] = rolling.sum().reset_index(level=0, drop=True) if group_by else rolling.sum()
                
                self.feature_names.append(feat_name)
        
        logger.info(f"Created rolling features for '{column}' with windows {windows}")
        return df_feat
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Categorical columns to encode
            method: Encoding method ('onehot', 'label', 'target')
            
        Returns:
            DataFrame with encoded features
        """
        df_feat = df.copy()
        
        if method == 'onehot':
            df_feat = pd.get_dummies(df_feat, columns=columns, prefix=columns, drop_first=True)
            logger.info(f"One-hot encoded {len(columns)} categorical columns")
        
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                if col in df_feat.columns:
                    le = LabelEncoder()
                    df_feat[col] = le.fit_transform(df_feat[col].astype(str))
                    self.transformers[f"{col}_label"] = le
            logger.info(f"Label encoded {len(columns)} categorical columns")
        
        return df_feat


class DataScaler:
    """
    Utility class for data normalization and scaling.
    """
    
    def __init__(self):
        """Initialize data scaler"""
        self.scalers = {}
    
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'standard',
        feature_range: Tuple[float, float] = (0, 1)
    ) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (None for all numeric)
            method: Scaling method ('standard', 'minmax', 'robust')
            feature_range: Range for minmax scaling
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.scalers[method] = scaler
        
        logger.info(f"Scaled {len(columns)} columns using {method} scaling")
        return df_scaled
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        norm: str = 'l2'
    ) -> pd.DataFrame:
        """
        Normalize features using L1 or L2 norm.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            norm: Normalization type ('l1', 'l2', 'max')
            
        Returns:
            DataFrame with normalized features
        """
        from sklearn.preprocessing import Normalizer
        
        df_norm = df.copy()
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        normalizer = Normalizer(norm=norm)
        df_norm[columns] = normalizer.fit_transform(df_norm[columns])
        self.scalers[f'normalize_{norm}'] = normalizer
        
        logger.info(f"Normalized {len(columns)} columns using {norm} norm")
        return df_norm


class DataValidator:
    """
    Utility class for data validation and quality checks.
    """
    
    def __init__(self):
        """Initialize data validator"""
        pass
    
    def generate_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataQualityReport object
        """
        # Basic info
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = {
            col: (count / total_rows * 100) if total_rows > 0 else 0
            for col, count in missing_values.items()
        }
        
        # Data types
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Outliers (using IQR method)
        outliers = {}
        cleaner = DataCleaner()
        outlier_masks = cleaner.detect_outliers(df, method='iqr')
        for col, mask in outlier_masks.items():
            outliers[col] = mask.sum()
        
        # Duplicates
        duplicate_rows = df.duplicated().sum()
        
        # Warnings
        warnings_list = []
        for col, pct in missing_percentage.items():
            if pct > 50:
                warnings_list.append(f"Column '{col}' has {pct:.1f}% missing values")
        
        if duplicate_rows > 0:
            warnings_list.append(f"Found {duplicate_rows} duplicate rows")
        
        for col, count in outliers.items():
            if count > total_rows * 0.05:  # More than 5% outliers
                warnings_list.append(f"Column '{col}' has {count} outliers ({count/total_rows*100:.1f}%)")
        
        report = DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            outliers=outliers,
            duplicate_rows=duplicate_rows,
            data_types=data_types,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            warnings=warnings_list
        )
        
        logger.info(f"Generated quality report for dataset with {total_rows} rows and {total_columns} columns")
        return report
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        expected_types: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema.
        
        Args:
            df: Input DataFrame
            expected_columns: List of expected column names
            expected_types: Dictionary of expected data types
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            errors.append(f"Unexpected columns: {extra_cols}")
        
        # Check data types
        if expected_types:
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if expected_type not in actual_type:
                        errors.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_ranges(
        self,
        df: pd.DataFrame,
        range_constraints: Dict[str, Tuple[float, float]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that numeric columns fall within expected ranges.
        
        Args:
            df: Input DataFrame
            range_constraints: Dictionary mapping columns to (min, max) tuples
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for col, (min_val, max_val) in range_constraints.items():
            if col not in df.columns:
                errors.append(f"Column '{col}' not found")
                continue
            
            if df[col].min() < min_val:
                errors.append(f"Column '{col}' has values below minimum {min_val}")
            
            if df[col].max() > max_val:
                errors.append(f"Column '{col}' has values above maximum {max_val}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def check_data_drift(
        self,
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 0.1
    ) -> Dict[str, Dict[str, float]]:
        """
        Check for data drift between reference and current datasets.
        
        Args:
            df_reference: Reference DataFrame
            df_current: Current DataFrame
            columns: Columns to check (None for all numeric)
            threshold: Drift threshold
            
        Returns:
            Dictionary with drift metrics per column
        """
        if columns is None:
            columns = df_reference.select_dtypes(include=[np.number]).columns.tolist()
        
        drift_report = {}
        
        for col in columns:
            if col not in df_reference.columns or col not in df_current.columns:
                continue
            
            ref_mean = df_reference[col].mean()
            curr_mean = df_current[col].mean()
            ref_std = df_reference[col].std()
            curr_std = df_current[col].std()
            
            mean_drift = abs(curr_mean - ref_mean) / (ref_mean + 1e-8)
            std_drift = abs(curr_std - ref_std) / (ref_std + 1e-8)
            
            drift_report[col] = {
                'mean_drift': mean_drift,
                'std_drift': std_drift,
                'has_drift': mean_drift > threshold or std_drift > threshold
            }
            
            if drift_report[col]['has_drift']:
                logger.warning(f"Data drift detected in column '{col}': mean_drift={mean_drift:.3f}, std_drift={std_drift:.3f}")
        
        return drift_report


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_strategy: str = 'mean',
    scaling_method: str = 'standard'
) -> Pipeline:
    """
    Create a scikit-learn preprocessing pipeline.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        numeric_strategy: Imputation strategy for numeric features
        scaling_method: Scaling method for numeric features
        
    Returns:
        Configured Pipeline object
    """
    from sklearn.preprocessing import OneHotEncoder
    
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler() if scaling_method == 'standard' else MinMaxScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    logger.info(f"Created preprocessing pipeline with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
    return pipeline


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 100],  # Has missing value and outlier
        'feature2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    print("Original data:")
    print(df)
    
    # Data cleaning
    cleaner = DataCleaner()
    df_clean = cleaner.handle_missing_values(df, strategy='mean')
    outliers = cleaner.detect_outliers(df_clean, method='iqr')
    df_clean = cleaner.handle_outliers(df_clean, outliers, strategy='clip')
    
    print("\nCleaned data:")
    print(df_clean)
    
    # Data validation
    validator = DataValidator()
    report = validator.generate_quality_report(df)
    print(f"\nQuality report: {report.warnings}")

"""
Data Processing Module with GPU Acceleration Support
Handles data loading, cleaning, and preprocessing with RAPIDS cuDF when available
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import logging
from loguru import logger
import streamlit as st
import io

# Try to import RAPIDS for GPU acceleration
try:
    import cudf
    import cupy as cp
    RAPIDS_AVAILABLE = True
    logger.info("RAPIDS cuDF available for GPU acceleration")
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.info("RAPIDS not available, using CPU processing")

class DataProcessor:
    """Advanced data processor with optional GPU acceleration"""
    
    def __init__(self, config=None):
        self.config = config
        self.use_gpu = config and config.is_gpu_available() if config else False
        self.logger = logger
        
        if self.use_gpu and RAPIDS_AVAILABLE:
            self.logger.info("GPU acceleration enabled with RAPIDS cuDF")
        else:
            self.logger.info("Using CPU-based data processing")
    
    def load_data(self, file_input: Union[str, Path]) -> pd.DataFrame:
        """Load data from various file formats with automatic format detection"""
        try:
            # Handle Streamlit file upload object
            if hasattr(file_input, 'name'):
                file_extension = Path(file_input.name).suffix.lower()
                
                # Reset file pointer to beginning
                if hasattr(file_input, 'seek'):
                    file_input.seek(0)
                
                if file_extension == '.csv':
                    df = pd.read_csv(file_input)
                elif file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_input)
                elif file_extension == '.json':
                    df = pd.read_json(file_input)
                elif file_extension == '.parquet':
                    df = pd.read_parquet(file_input)
                else:
                    # Try to detect format from content
                    df = self._auto_detect_format(file_input)
                    
            # Handle file path string
            else:
                file_path = Path(file_input)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_input}")
                    
                file_extension = file_path.suffix.lower()
                
                if file_extension == '.csv':
                    if self.use_gpu and RAPIDS_AVAILABLE:
                        df = cudf.read_csv(file_path).to_pandas()
                    else:
                        df = pd.read_csv(file_path)
                elif file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif file_extension == '.json':
                    df = pd.read_json(file_path)
                elif file_extension == '.parquet':
                    if self.use_gpu and RAPIDS_AVAILABLE:
                        df = cudf.read_parquet(file_path).to_pandas()
                    else:
                        df = pd.read_parquet(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data validation
            if df.empty:
                raise ValueError("Loaded dataset is empty")
                
            self.logger.info(f"✅ Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def _auto_detect_format(self, file_input):
        """Auto-detect file format from content"""
        try:
            # Read first few bytes to detect format
            if hasattr(file_input, 'read'):
                content_start = file_input.read(1024)
                file_input.seek(0)
                
                if b'parquet' in content_start.lower():
                    return pd.read_parquet(file_input)
                elif b'{' in content_start:
                    return pd.read_json(file_input)
                else:
                    # Default to CSV
                    return pd.read_csv(file_input)
            else:
                return pd.read_csv(file_input)
        except:
            # Final fallback
            return pd.read_csv(file_input)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        try:
            analysis = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'dtypes': df.dtypes.value_counts().to_dict()
                },
                'missing_data': {
                    'total_missing': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'columns_with_missing': df.columns[df.isnull().any()].tolist(),
                    'missing_by_column': df.isnull().sum().to_dict()
                },
                'duplicates': {
                    'duplicate_rows': df.duplicated().sum(),
                    'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
                },
                'data_types': {
                    'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                    'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                    'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
                }
            }
            
            # Calculate data quality score
            missing_penalty = min(analysis['missing_data']['missing_percentage'] * 0.01, 0.3)
            duplicate_penalty = min(analysis['duplicates']['duplicate_percentage'] * 0.01, 0.2)
            
            analysis['quality_score'] = max(0.0, 1.0 - missing_penalty - duplicate_penalty)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_quality_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing data quality: {str(e)}")
            return {}
    
    def preprocess_data(self, 
                       df: pd.DataFrame,
                       handle_missing: str = 'auto',
                       encode_categorical: bool = True,
                       normalize_features: bool = False,
                       remove_outliers: bool = False,
                       feature_selection: bool = False) -> pd.DataFrame:
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            df: Input dataframe
            handle_missing: Method to handle missing values
            encode_categorical: Whether to encode categorical variables
            normalize_features: Whether to normalize numerical features
            remove_outliers: Whether to remove statistical outliers
            feature_selection: Whether to perform automatic feature selection
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            self.logger.info(f"Starting preprocessing - Original shape: {df.shape}")
            self.logger.info(f"Options: missing={handle_missing}, encode={encode_categorical}, normalize={normalize_features}, outliers={remove_outliers}")
            
            # Validate input
            if df is None or df.empty:
                raise ValueError("Input dataframe is empty or None")
            
            # Make a copy to avoid modifying original
            df_copy = df.copy()
            
            if self.use_gpu and RAPIDS_AVAILABLE:
                processed_df = self._preprocess_gpu(df_copy, handle_missing, encode_categorical, 
                                                  normalize_features, remove_outliers)
            else:
                processed_df = self._preprocess_cpu(df_copy, handle_missing, encode_categorical,
                                                  normalize_features, remove_outliers)
            
            # Validate output
            if processed_df is None:
                raise ValueError("Preprocessing returned None")
            if processed_df.empty:
                self.logger.warning("Preprocessing resulted in empty dataframe")
            
            self.logger.info(f"✅ Preprocessing completed - Final shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in data preprocessing: {str(e)}")
            self.logger.info("Returning original dataframe due to preprocessing error")
            return df.copy() if df is not None else pd.DataFrame()
    
    def _preprocess_gpu(self, df, handle_missing, encode_categorical, 
                       normalize_features, remove_outliers):
        """GPU-accelerated preprocessing using cuDF"""
        
        self.logger.info("Using GPU-accelerated preprocessing")
        
        try:
            # Convert to cuDF for processing
            gdf = cudf.from_pandas(df)
            
            # Handle missing values
            if handle_missing == 'drop_rows':
                original_rows = len(gdf)
                gdf = gdf.dropna()
                self.logger.info(f"Dropped rows with missing values: {original_rows} → {len(gdf)}")
                
            elif handle_missing == 'drop_columns':
                # Drop columns with more than 50% missing values
                missing_threshold = len(gdf) * 0.5
                cols_to_keep = gdf.columns[gdf.isnull().sum() < missing_threshold]
                original_cols = len(gdf.columns)
                gdf = gdf[cols_to_keep]
                self.logger.info(f"Dropped columns with >50% missing: {original_cols} → {len(gdf.columns)}")
                
            elif handle_missing == 'fill_median':
                numeric_cols = gdf.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if gdf[col].isnull().any():
                        gdf[col] = gdf[col].fillna(gdf[col].median())
                self.logger.info(f"Filled missing values with median for {len(numeric_cols)} columns")
                
            elif handle_missing == 'fill_mean':
                numeric_cols = gdf.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if gdf[col].isnull().any():
                        gdf[col] = gdf[col].fillna(gdf[col].mean())
                self.logger.info(f"Filled missing values with mean for {len(numeric_cols)} columns")
                
            elif handle_missing == 'fill_mode':
                for col in gdf.columns:
                    if gdf[col].isnull().any():
                        mode_val = gdf[col].mode()
                        if len(mode_val) > 0:
                            gdf[col] = gdf[col].fillna(mode_val[0])
                self.logger.info(f"Filled missing values with mode")
                
            elif handle_missing == 'auto':
                # Smart missing value handling
                filled_cols = 0
                for col in gdf.columns:
                    if gdf[col].isnull().any():
                        if gdf[col].dtype in ['object', 'category']:
                            mode_val = gdf[col].mode()
                            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                            gdf[col] = gdf[col].fillna(fill_val)
                            filled_cols += 1
                        else:
                            gdf[col] = gdf[col].fillna(gdf[col].median())
                            filled_cols += 1
                self.logger.info(f"Auto-filled missing values for {filled_cols} columns")
            
            # Encode categorical variables
            if encode_categorical:
                categorical_cols = gdf.select_dtypes(include=['object']).columns
                encoded_cols = 0
                for col in categorical_cols:
                    if col in gdf.columns:  # Check if column still exists
                        gdf[col] = gdf[col].astype('category').cat.codes
                        encoded_cols += 1
                self.logger.info(f"Encoded {encoded_cols} categorical columns")
            
            # Normalize features
            if normalize_features:
                numeric_cols = gdf.select_dtypes(include=['number']).columns
                normalized_cols = 0
                for col in numeric_cols:
                    if col in gdf.columns:  # Check if column still exists
                        col_mean = gdf[col].mean()
                        col_std = gdf[col].std()
                        if col_std > 0:
                            gdf[col] = (gdf[col] - col_mean) / col_std
                            normalized_cols += 1
                self.logger.info(f"Normalized {normalized_cols} numeric columns")
            
            # Remove outliers using IQR method
            if remove_outliers:
                numeric_cols = gdf.select_dtypes(include=['number']).columns
                original_rows = len(gdf)
                for col in numeric_cols:
                    if col in gdf.columns:  # Check if column still exists
                        Q1 = gdf[col].quantile(0.25)
                        Q3 = gdf[col].quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:  # Only remove outliers if IQR is meaningful
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            gdf = gdf[(gdf[col] >= lower_bound) & (gdf[col] <= upper_bound)]
                self.logger.info(f"Removed outliers: {original_rows} → {len(gdf)} rows")
            
            return gdf.to_pandas() if len(gdf) > 0 else df
            
        except Exception as e:
            self.logger.error(f"❌ Error in GPU preprocessing: {str(e)}")
            raise
    
    def _preprocess_cpu(self, df, handle_missing, encode_categorical,
                       normalize_features, remove_outliers):
        """CPU-based preprocessing fallback"""
        
        self.logger.info("Using CPU-based preprocessing")
        
        try:
            df = df.copy()
            
            # Handle missing values
            if handle_missing == 'drop_rows':
                original_rows = len(df)
                df = df.dropna()
                self.logger.info(f"Dropped rows with missing values: {original_rows} → {len(df)}")
                
            elif handle_missing == 'drop_columns':
                missing_threshold = len(df) * 0.5
                cols_to_keep = df.columns[df.isnull().sum() < missing_threshold]
                original_cols = len(df.columns)
                df = df[cols_to_keep]
                self.logger.info(f"Dropped columns with >50% missing: {original_cols} → {len(df.columns)}")
                
            elif handle_missing == 'fill_median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median())
                self.logger.info(f"Filled missing values with median for {len(numeric_cols)} columns")
                
            elif handle_missing == 'fill_mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mean())
                self.logger.info(f"Filled missing values with mean for {len(numeric_cols)} columns")
                
            elif handle_missing == 'fill_mode':
                filled_cols = 0
                for col in df.columns:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
                            filled_cols += 1
                self.logger.info(f"Filled missing values with mode for {filled_cols} columns")
                
            elif handle_missing == 'auto':
                filled_cols = 0
                for col in df.columns:
                    if df[col].isnull().any():
                        if df[col].dtype == 'object':
                            mode_val = df[col].mode()
                            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                            df[col] = df[col].fillna(fill_val)
                            filled_cols += 1
                        else:
                            df[col] = df[col].fillna(df[col].median())
                            filled_cols += 1
                self.logger.info(f"Auto-filled missing values for {filled_cols} columns")
            
            # Encode categorical variables
            if encode_categorical:
                from sklearn.preprocessing import LabelEncoder
                categorical_cols = df.select_dtypes(include=['object']).columns
                encoded_cols = 0
                for col in categorical_cols:
                    if col in df.columns:  # Check if column still exists
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        encoded_cols += 1
                self.logger.info(f"Encoded {encoded_cols} categorical columns")
            
            # Normalize features
            if normalize_features:
                from sklearn.preprocessing import StandardScaler
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    self.logger.info(f"Normalized {len(numeric_cols)} numeric columns")
            
            # Remove outliers
            if remove_outliers:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                original_rows = len(df)
                for col in numeric_cols:
                    if col in df.columns:  # Check if column still exists
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:  # Only remove outliers if IQR is meaningful
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                self.logger.info(f"Removed outliers: {original_rows} → {len(df)} rows")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error in CPU preprocessing: {str(e)}")
            raise
    
    
    def _generate_quality_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate data quality recommendations"""
        
        recommendations = []
        
        missing_pct = analysis['missing_data']['missing_percentage']
        duplicate_pct = analysis['duplicates']['duplicate_percentage']
        
        if missing_pct > 20:
            recommendations.append(f"High missing values ({missing_pct:.1f}%) - consider advanced imputation")
        elif missing_pct > 5:
            recommendations.append(f"Moderate missing values ({missing_pct:.1f}%) - handle before modeling")
        
        if duplicate_pct > 10:
            recommendations.append(f"High duplicate rows ({duplicate_pct:.1f}%) - remove duplicates")
        elif duplicate_pct > 1:
            recommendations.append(f"Some duplicate rows ({duplicate_pct:.1f}%) - investigate and remove")
        
        if len(analysis['data_types']['categorical_columns']) > 0:
            recommendations.append("Categorical variables found - encode before modeling")
        
        numeric_cols = len(analysis['data_types']['numeric_columns'])
        if numeric_cols > 10:
            recommendations.append(f"Many features ({numeric_cols}) - consider feature selection")
        
        if analysis['basic_info']['memory_usage_mb'] > 1000:
            recommendations.append("Large dataset - consider using data sampling or chunked processing")
        
        return recommendations
    
    def create_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive data profile"""
        
        try:
            profile = {
                'overview': {
                    'dataset_size': len(df),
                    'feature_count': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'data_types': df.dtypes.value_counts().to_dict()
                },
                'features': {},
                'quality_issues': [],
                'suggestions': []
            }
            
            # Analyze each column
            for col in df.columns:
                col_profile = {
                    'dtype': str(df[col].dtype),
                    'missing_count': df[col].isnull().sum(),
                    'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                    'unique_count': df[col].nunique(),
                    'unique_percentage': (df[col].nunique() / len(df)) * 100
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_profile.update({
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'median': df[col].median(),
                        'std': df[col].std(),
                        'skewness': df[col].skew() if len(df) > 1 else 0,
                        'outliers_count': self._count_outliers(df[col])
                    })
                else:
                    top_values = df[col].value_counts().head().to_dict()
                    col_profile['top_values'] = top_values
                
                profile['features'][col] = col_profile
            
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Error creating data profile: {str(e)}")
            return {}
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return len(outliers)
        except:
            return 0
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """Save processed data to file"""
        
        try:
            output_path = Path(f"data/processed/{filename}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format.lower() == 'excel':
                df.to_excel(output_path, index=False)
            
            self.logger.info(f"✅ Saved processed data to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving processed data: {str(e)}")
            return None


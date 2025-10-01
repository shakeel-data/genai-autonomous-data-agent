"""
Exploratory Data Analysis Module
Advanced EDA with interactive visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class EDAModule:
    """Advanced Exploratory Data Analysis with interactive visualizations"""
    
    def __init__(self, config=None):
        self.config = config
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def generate_comprehensive_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive EDA report with better error handling"""
        
        try:
            print("Starting comprehensive EDA analysis...")
            eda_results = {}
            
            # Execute each analysis component with individual error handling
            try:
                print("Generating statistical summary...")
                eda_results['statistical_summary'] = self.statistical_summary(df)
            except Exception as e:
                print(f"Error in statistical summary: {str(e)}")
                eda_results['statistical_summary'] = pd.DataFrame()
            
            try:
                print("Generating data overview...")
                eda_results['data_overview'] = self.data_overview(df)
            except Exception as e:
                print(f"❌ Error in data overview: {str(e)}")
                eda_results['data_overview'] = {}
            
            try:
                print("Analyzing missing values...")
                eda_results['missing_values_analysis'] = self.analyze_missing_values(df)
            except Exception as e:
                print(f"❌ Error in missing values analysis: {str(e)}")
                eda_results['missing_values_analysis'] = {}
            
            try:
                print("Running correlation analysis...")
                eda_results['correlation_analysis'] = self.correlation_analysis(df)
            except Exception as e:
                print(f"❌ Error in correlation analysis: {str(e)}")
                eda_results['correlation_analysis'] = {}
            
            try:
                print("Analyzing distributions...")
                eda_results['distribution_analysis'] = self.analyze_distributions(df)
            except Exception as e:
                print(f"❌ Error in distribution analysis: {str(e)}")
                eda_results['distribution_analysis'] = {}
            
            try:
                print("Analyzing categorical variables...")
                eda_results['categorical_analysis'] = self.analyze_categorical_variables(df)
            except Exception as e:
                print(f"❌ Error in categorical analysis: {str(e)}")
                eda_results['categorical_analysis'] = {}
            
            try:
                print("Detecting outliers...")
                eda_results['outlier_analysis'] = self.detect_outliers(df)
            except Exception as e:
                print(f"❌ Error in outlier detection: {str(e)}")
                eda_results['outlier_analysis'] = {}
            
            try:
                print("Generating insights...")
                eda_results['insights'] = self.generate_insights(df)
            except Exception as e:
                print(f"❌ Error generating insights: {str(e)}")
                eda_results['insights'] = []
            
            print("✅ Comprehensive EDA completed successfully!")
            return eda_results
            
        except Exception as e:
            print(f"❌ Critical error in comprehensive EDA: {str(e)}")
            return {'error': str(e), 'insights': ['Error generating EDA results']}

    def statistical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced statistical summary"""
        
        try:
            # Basic statistics
            summary = df.describe(include='all').T
            
            # Add additional statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                summary.loc[col, 'missing_count'] = df[col].isnull().sum()
                summary.loc[col, 'missing_percentage'] = (df[col].isnull().sum() / len(df)) * 100
                summary.loc[col, 'skewness'] = df[col].skew()
                summary.loc[col, 'kurtosis'] = df[col].kurtosis()
                summary.loc[col, 'range'] = df[col].max() - df[col].min()
                
                # Outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                summary.loc[col, 'outliers_count'] = len(outliers)
            
            # Add statistics for categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                summary.loc[col, 'missing_count'] = df[col].isnull().sum()
                summary.loc[col, 'missing_percentage'] = (df[col].isnull().sum() / len(df)) * 100
                summary.loc[col, 'unique_values'] = df[col].nunique()
                summary.loc[col, 'most_frequent'] = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
                summary.loc[col, 'most_frequent_count'] = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating statistical summary: {str(e)}")
            return pd.DataFrame()
    
    def data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data overview metrics"""
        
        try:
            overview = {
                'basic_info': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'duplicate_rows': df.duplicated().sum()
                },
                'data_types': {
                    'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                    'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
                    'boolean_columns': len(df.select_dtypes(include=['bool']).columns)
                },
                'missing_data': {
                    'total_missing_values': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'columns_with_missing': df.columns[df.isnull().any()].tolist()
                }
            }
            
            return overview
            
        except Exception as e:
            print(f"❌ Error generating data overview: {str(e)}")
            return {}
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return {'message': 'Not enough numeric columns for correlation analysis'}
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Threshold for high correlation
                        high_corr_pairs.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            return {
                'correlation_matrix': corr_matrix,
                'high_correlations': high_corr_pairs,
                'avg_correlation': corr_matrix.mean().mean(),
                'max_correlation': corr_matrix.max().max(),
                'min_correlation': corr_matrix.min().min()
            }
            
        except Exception as e:
            print(f"❌ Error in correlation analysis: {str(e)}")
            return {}
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive missing values analysis"""
        
        try:
            missing_info = {}
            
            # Missing values per column
            missing_by_column = df.isnull().sum()
            missing_percentage = (missing_by_column / len(df)) * 100
            
            missing_info['by_column'] = pd.DataFrame({
                'missing_count': missing_by_column,
                'missing_percentage': missing_percentage
            }).sort_values('missing_count', ascending=False)
            
            # Missing values patterns
            missing_patterns = df.isnull().value_counts().head(10)
            missing_info['patterns'] = missing_patterns
            
            # Columns with high missing values
            high_missing = missing_info['by_column'][missing_info['by_column']['missing_percentage'] > 20]
            missing_info['high_missing_columns'] = high_missing.index.tolist()
            
            return missing_info
            
        except Exception as e:
            print(f"❌ Error analyzing missing values: {str(e)}")
            return {}
    
    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            distribution_info = {}
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                # Basic distribution statistics
                dist_stats = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'mode': col_data.mode()[0] if len(col_data.mode()) > 0 else np.nan,
                    'std': col_data.std(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'range': col_data.max() - col_data.min(),
                    'iqr': col_data.quantile(0.75) - col_data.quantile(0.25)
                }
                
                # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
                if len(col_data) <= 5000:
                    stat, p_value = stats.shapiro(col_data.sample(min(len(col_data), 5000)))
                    dist_stats['normality_test'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
                
                distribution_info[col] = dist_stats
            
            return distribution_info
            
        except Exception as e:
            print(f"❌ Error analyzing distributions: {str(e)}")
            return {}
    
    def analyze_categorical_variables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical variables"""
        
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            categorical_info = {}
            
            for col in categorical_cols:
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                value_counts = col_data.value_counts()
                
                cat_stats = {
                    'unique_count': col_data.nunique(),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'most_frequent_percentage': (value_counts.iloc[0] / len(col_data)) * 100 if len(value_counts) > 0 else 0,
                    'top_categories': value_counts.head(10).to_dict(),
                    'cardinality_level': self._categorize_cardinality(col_data.nunique(), len(col_data))
                }
                
                categorical_info[col] = cat_stats
            
            return categorical_info
            
        except Exception as e:
            print(f"❌ Error analyzing categorical variables: {str(e)}")
            return {}
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric variables using multiple methods"""
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {'message': 'No numeric columns found for outlier detection'}
            
            outlier_info = {}
            
            for col in numeric_cols:
                try:
                    col_data = df[col].dropna()
                    
                    if len(col_data) == 0:
                        outlier_info[col] = {
                            'iqr_outliers': 0,
                            'iqr_percentage': 0,
                            'zscore_outliers': 0,
                            'zscore_percentage': 0,
                            'modified_zscore_outliers': 0,
                            'modified_zscore_percentage': 0,
                            'outlier_values_sample': [],
                            'error': 'No valid data'
                        }
                        continue
                    
                    # IQR method
                    try:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR > 0:  # Avoid division by zero
                            iqr_outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                            iqr_count = len(iqr_outliers)
                            iqr_pct = (iqr_count / len(col_data)) * 100
                        else:
                            iqr_outliers = pd.Series(dtype=float)
                            iqr_count = 0
                            iqr_pct = 0
                    except Exception:
                        iqr_outliers = pd.Series(dtype=float)
                        iqr_count = 0
                        iqr_pct = 0
                    
                    # Z-score method (handle edge cases)
                    try:
                        if col_data.std() > 0:
                            z_scores = np.abs(stats.zscore(col_data))
                            zscore_outliers = col_data[z_scores > 3]
                            zscore_count = len(zscore_outliers)
                            zscore_pct = (zscore_count / len(col_data)) * 100
                        else:
                            zscore_outliers = pd.Series(dtype=float)
                            zscore_count = 0
                            zscore_pct = 0
                    except Exception:
                        zscore_outliers = pd.Series(dtype=float)
                        zscore_count = 0
                        zscore_pct = 0
                    
                    # Modified Z-score method
                    try:
                        median = col_data.median()
                        mad = np.median(np.abs(col_data - median))
                        
                        if mad > 0:
                            modified_z_scores = 0.6745 * (col_data - median) / mad
                            modified_zscore_outliers = col_data[np.abs(modified_z_scores) > 3.5]
                            modified_zscore_count = len(modified_zscore_outliers)
                            modified_zscore_pct = (modified_zscore_count / len(col_data)) * 100
                        else:
                            modified_zscore_outliers = pd.Series(dtype=float)
                            modified_zscore_count = 0
                            modified_zscore_pct = 0
                    except Exception:
                        modified_zscore_outliers = pd.Series(dtype=float)
                        modified_zscore_count = 0
                        modified_zscore_pct = 0
                    
                    # Store results
                    outlier_info[col] = {
                        'iqr_outliers': iqr_count,
                        'iqr_percentage': round(iqr_pct, 2),
                        'zscore_outliers': zscore_count,
                        'zscore_percentage': round(zscore_pct, 2),
                        'modified_zscore_outliers': modified_zscore_count,
                        'modified_zscore_percentage': round(modified_zscore_pct, 2),
                        'outlier_values_sample': iqr_outliers.head(5).tolist() if len(iqr_outliers) > 0 else []
                    }
                    
                except Exception as e:
                    print(f"❌ Error processing column {col}: {str(e)}")
                    outlier_info[col] = {
                        'iqr_outliers': 0,
                        'iqr_percentage': 0,
                        'zscore_outliers': 0,
                        'zscore_percentage': 0,
                        'modified_zscore_outliers': 0,
                        'modified_zscore_percentage': 0,
                        'outlier_values_sample': [],
                        'error': str(e)
                    }
            
            return outlier_info
            
        except Exception as e:
            print(f"❌ Error in outlier detection: {str(e)}")
            return {'error': str(e)}

    def generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate automated insights from data analysis"""
        
        insights = []
        
        try:
            # Data size insights
            if len(df) > 100000:
                insights.append(f"Large dataset with {len(df):,} rows - consider sampling for faster analysis")
            elif len(df) < 100:
                insights.append(f"Small dataset with {len(df)} rows - results may not be statistically significant")
            
            # Missing values insights
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 30:
                insights.append(f"High missing values ({missing_pct:.1f}%) - data quality may be poor")
            elif missing_pct > 10:
                insights.append(f"Moderate missing values ({missing_pct:.1f}%) - consider imputation strategies")
            
            # Duplicate insights
            duplicate_pct = (df.duplicated().sum() / len(df)) * 100
            if duplicate_pct > 10:
                insights.append(f"High duplicate rows ({duplicate_pct:.1f}%) - may indicate data quality issues")
            
            # Correlation insights
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr_matrix = numeric_df.corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.8:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                if high_corr_pairs:
                    insights.append(f"Found {len(high_corr_pairs)} highly correlated variable pairs - consider feature selection")
            
            # Cardinality insights
            categorical_cols = df.select_dtypes(include=['object']).columns
            high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.5]
            if high_cardinality_cols:
                insights.append(f"High cardinality categorical variables found: {', '.join(high_cardinality_cols)}")
            
            # Skewness insights
            for col in numeric_df.columns:
                skewness = df[col].skew()
                if abs(skewness) > 2:
                    insights.append(f"Variable '{col}' is highly skewed ({skewness:.2f}) - consider transformation")
            
            # Imbalanced data insights
            for col in categorical_cols[:5]:  # Check first 5 categorical columns
                if df[col].nunique() <= 10:  # Only for low cardinality
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 1:
                        imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                        if imbalance_ratio > 10:
                            insights.append(f"Highly imbalanced categorical variable '{col}' - most frequent value is {imbalance_ratio:.1f}x more common")
            
            return insights
            
        except Exception as e:
            print(f"❌ Error generating insights: {str(e)}")
            return ["Error generating insights from the data"]
    
    def _categorize_cardinality(self, unique_count: int, total_count: int) -> str:
        """Categorize cardinality level"""
        ratio = unique_count / total_count
        if ratio > 0.9:
            return "Very High"
        elif ratio > 0.5:
            return "High"
        elif ratio > 0.1:
            return "Medium"
        else:
            return "Low"
    
    # Visualization methods
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive correlation heatmap"""
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return go.Figure().add_annotation(text="Not enough numeric columns for correlation heatmap")
            
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.around(corr_matrix.values, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Correlation Heatmap",
                width=800,
                height=600,
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create distribution plot for a numeric column"""
        
        try:
            if column not in df.columns:
                return go.Figure().add_annotation(text=f"Column '{column}' not found")
            
            if df[column].dtype not in ['int64', 'float64']:
                return go.Figure().add_annotation(text=f"Column '{column}' is not numeric")
            
            data = df[column].dropna()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Histogram", "Box Plot", "QQ Plot", "Distribution Statistics"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=data, nbinsx=30, name="Distribution", showlegend=False),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=data, name=column, showlegend=False),
                row=1, col=2
            )
            
            # QQ plot
            qq_data = stats.probplot(data, dist="norm")
            fig.add_trace(
                go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name='QQ Plot',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add reference line for QQ plot
            fig.add_trace(
                go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red'),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Statistics table
            stats_data = [
                ['Mean', f"{data.mean():.3f}"],
                ['Median', f"{data.median():.3f}"],
                ['Std Dev', f"{data.std():.3f}"],
                ['Skewness', f"{data.skew():.3f}"],
                ['Kurtosis', f"{data.kurtosis():.3f}"],
                ['Min', f"{data.min():.3f}"],
                ['Max', f"{data.max():.3f}"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=["Statistic", "Value"]),
                    cells=dict(values=[[row[0] for row in stats_data], [row[1] for row in stats_data]])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f"Distribution Analysis: {column}",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating distribution plot: {str(e)}")
            return go.Figure()
    
    def create_missing_values_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create missing values visualization"""
        
        try:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
            
            if len(missing_data) == 0:
                return go.Figure().add_annotation(text="No missing values found in the dataset")
            
            missing_percentage = (missing_data / len(df)) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=missing_data.index,
                x=missing_percentage.values,
                orientation='h',
                marker_color='coral',
                text=[f"{count} ({pct:.1f}%)" for count, pct in zip(missing_data.values, missing_percentage.values)],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Missing Values Analysis",
                xaxis_title="Missing Percentage (%)",
                yaxis_title="Columns",
                height=max(400, len(missing_data) * 30),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating missing values plot: {str(e)}")
            return go.Figure()

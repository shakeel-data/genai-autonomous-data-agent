# 📚 API Documentation - GenAI Autonomous Data Agent

## Overview

This document provides comprehensive API documentation for all modules in the GenAI Autonomous Data Agent platform.

## Core Modules

### 📊 DataProcessor

**Class**: `DataProcessor(config=None)`

**Methods**:

#### `load_data(file_input)`
Load data from various file formats.

**Parameters**:
- `file_input`: File path or uploaded file object

**Returns**: `pd.DataFrame`

**Example**:
processor = DataProcessor()
df = processor.load_data("data.csv")


#### `analyze_data_quality(df)`
Perform comprehensive data quality analysis.

**Parameters**:
- `df`: Input DataFrame

**Returns**: `Dict[str, Any]` - Quality analysis results

#### `preprocess_data(df, **kwargs)`
Preprocess data with various options.

**Parameters**:
- `df`: Input DataFrame
- `handle_missing`: Method for missing values ('auto', 'drop', 'fill_median', etc.)
- `encode_categorical`: Boolean, encode categorical variables
- `normalize_features`: Boolean, normalize numerical features
- `remove_outliers`: Boolean, remove statistical outliers

**Returns**: `pd.DataFrame` - Preprocessed data

### 🔍 EDAModule

**Class**: `EDAModule(config=None)`

#### `generate_comprehensive_eda(df)`
Generate complete EDA analysis.

**Parameters**:
- `df`: Input DataFrame

**Returns**: `Dict[str, Any]` - EDA results

#### `create_correlation_heatmap(df)`
Create interactive correlation heatmap.

**Returns**: `plotly.graph_objects.Figure`

#### `generate_insights(df)`
Generate AI-powered insights.

**Returns**: `List[str]` - List of insights

### 🤖 MLModule

**Class**: `MLModule(config=None)`

#### `auto_ml_pipeline(X, y, **kwargs)`
Complete automated ML pipeline.

**Parameters**:
- `X`: Features DataFrame
- `y`: Target Series
- `task_type`: 'auto', 'classification', or 'regression'
- `test_size`: Float, proportion for test set
- `cv_folds`: Integer, cross-validation folds

**Returns**: `Dict[str, Any]` - ML results

#### `predict(model_name, X)`
Make predictions using trained model.

**Parameters**:
- `model_name`: String, name of trained model
- `X`: Features DataFrame for prediction

**Returns**: `numpy.ndarray` - Predictions

### 💡 ExplainModule

**Class**: `ExplainModule(config=None)`

#### `create_comprehensive_explanation(model, X, **kwargs)`
Generate comprehensive model explanations.

**Parameters**:
- `model`: Trained ML model
- `X`: Features DataFrame
- `task_type`: 'classification' or 'regression'
- `sample_size`: Integer, samples for analysis

**Returns**: `Dict[str, Any]` - Explanation results

### 💬 SQLAgent

**Class**: `SQLAgent(config=None)`

#### `natural_language_to_sql(user_query, df, table_name)`
Convert natural language to SQL.

**Parameters**:
- `user_query`: String, natural language query
- `df`: Source DataFrame
- `table_name`: String, table name in database

**Returns**: `str` - Generated SQL query

#### `execute_sql_query(sql_query, max_rows)`
Execute SQL query and return results.

**Parameters**:
- `sql_query`: String, SQL query to execute
- `max_rows`: Integer, maximum rows to return

**Returns**: `pd.DataFrame` - Query results

### 📋 ReportGenerator

**Class**: `ReportGenerator(config=None)`

#### `generate_comprehensive_report(data, **kwargs)`
Generate comprehensive analysis report.

**Parameters**:
- `data`: Dict containing analysis results
- `report_title`: String, title of report
- `format`: String, output format ('html', 'pdf', 'docx')
- `include_sections`: Dict specifying sections to include

**Returns**: `str` - Path to generated report

## Configuration

### Config Class

**File**: `config.yaml`

**Key Sections**:
- `app`: Application settings
- `gpu`: GPU acceleration settings
- `database`: Database configuration
- `ml`: Machine learning parameters
- `visualization`: Chart and plot settings

## Error Handling

All methods include comprehensive error handling:
- Input validation
- Graceful fallbacks for missing dependencies
- Detailed logging of errors and warnings

## Examples

### Complete Workflow Example
from src.config import Config
from src.data_processor import DataProcessor
from src.ml_module import MLModule
from src.explain_module import ExplainModule

Initialize
config = Config()
processor = DataProcessor(config)
ml_module = MLModule(config)
explainer = ExplainModule(config)

Load and process data
df = processor.load_data("data.csv")
processed_df = processor.preprocess_data(df)

Train models
X = processed_df.drop('target', axis=1)
y = processed_df['target']
ml_results = ml_module.auto_ml_pipeline(X, y)

Generate explanations
best_model = ml_results['best_model_object']
explanations = explainer.create_comprehensive_explanation(best_model, X)


## Dependencies

### Required Packages
- `pandas>=2.1.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `streamlit>=1.28.0`
- `plotly>=5.17.0`

### Optional GPU Packages
- `cudf>=24.08.0` (RAPIDS)
- `cuml>=24.08.0` (RAPIDS)

### Explainability Packages
- `shap>=0.43.0`
- `lime>=0.2.0`

## Performance Notes

- GPU acceleration available with NVIDIA RAPIDS
- Automatic fallback to CPU processing
- Optimized for datasets up to 10M rows
- Memory-efficient processing for large datasets

## Support

For issues and questions:
- Check logs in `logs/` directory
- Review error messages for troubleshooting
- Consult user manual for common solutions

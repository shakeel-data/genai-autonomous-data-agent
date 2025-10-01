# 👤 User Manual - GenAI Autonomous Data Agent

## 🚀 Getting Started

### System Requirements

**Minimum Requirements**:
- Python 3.8+
- 8GB RAM
- 2GB free disk space

**Recommended for GPU Acceleration**:
- NVIDIA GPU with CUDA support
- 16GB+ RAM
- NVIDIA RAPIDS compatible GPU

### Installation

1. **Clone Repository**:
git clone https://github.com/your-repo/genai-autonomous-data-agent.git
cd genai-autonomous-data-agent


2. **Setup Environment**:
python -m venv genai_agent
source genai_agent/bin/activate # Linux/Mac

or genai_agent\Scripts\activate # Windows


3. **Install Dependencies**:
pip install -r requirements.txt

4. **Run Setup**:
python setup.py


5. **Launch Application**:
streamlit run app.py


## 📊 Using the Platform

### Step 1: Data Upload

1. **Navigate to "Data Upload & Processing"**
2. **Choose Upload Method**:
   - Upload your CSV/Excel file
   - Or select a sample dataset
3. **Review Data Quality**:
   - Check statistics and quality score
   - Review recommendations
4. **Apply Preprocessing**:
   - Configure preprocessing options
   - Click "Apply Preprocessing"

### Step 2: Exploratory Data Analysis

1. **Navigate to "Exploratory Data Analysis"**
2. **Select Analysis Components**:
   - Statistical Summary
   - Correlation Analysis
   - Distribution Analysis
   - Missing Values Analysis
   - Outlier Detection
   - Automated Insights
3. **Run Analysis**:
   - Click "Run Complete EDA"
   - Review generated insights and visualizations

### Step 3: Machine Learning Training

1. **Navigate to "Machine Learning Training"**
2. **Configure Model**:
   - Select target variable
   - Choose features (or use auto-selection)
   - Set task type (auto/classification/regression)
3. **Set Advanced Options**:
   - Test set size
   - Cross-validation folds
   - Random state
4. **Train Models**:
   - Click "Train Models with Auto-ML"
   - Review performance comparison
   - Identify best model

### Step 4: Model Explainability

1. **Navigate to "Model Explainability"**
2. **Select Model**: Choose model to explain
3. **Choose Methods**:
   - SHAP Analysis (global explanations)
   - LIME Analysis (local explanations)  
   - Feature Importance
   - Business Insights
4. **Generate Explanations**:
   - Set sample size
   - Click "Generate Explanations"
   - Review visualizations and insights

### Step 5: SQL Natural Language Chat

1. **Navigate to "SQL Natural Language Chat"**
2. **Ask Questions**: Type questions in plain English
3. **Review Results**:
   - Generated SQL query
   - Query explanation
   - Results table
   - Suggested visualizations
4. **Query History**: Access previous queries

### Step 6: Report Generation

1. **Navigate to "Automated Report Generation"**
2. **Configure Report**:
   - Set report title
   - Choose output format (HTML/PDF/DOCX)
   - Select sections to include
3. **Generate Report**:
   - Click "Generate Complete Report"
   - Download generated file

## 💡 Tips & Best Practices

### Data Upload Tips
- **File Size**: Keep files under 500MB for optimal performance
- **Data Quality**: Clean data produces better results
- **Column Names**: Use descriptive, consistent column names
- **Missing Values**: Handle missing values appropriately for your use case

### EDA Best Practices
- **Start with Overview**: Always review basic statistics first
- **Check Correlations**: Identify highly correlated features
- **Understand Distributions**: Look for skewed or unusual distributions
- **Missing Values**: Understand patterns in missing data

### ML Training Tips
- **Target Selection**: Choose clear, well-defined target variables
- **Feature Selection**: Remove irrelevant or redundant features
- **Task Type**: Let auto-detection choose unless you're certain
- **Validation**: Use appropriate train/test split ratios

### Explainability Best Practices
- **Sample Size**: Use smaller samples (100-500) for faster processing
- **Multiple Methods**: Compare SHAP and LIME results
- **Business Context**: Interpret results in business context
- **Feature Names**: Use meaningful feature names for clarity

### SQL Chat Tips
- **Be Specific**: Clear, specific questions get better results
- **Use Column Names**: Reference actual column names when possible
- **Start Simple**: Begin with basic queries and build complexity
- **Review SQL**: Always check generated SQL before trusting results

## 🔧 Troubleshooting

### Common Issues

#### "GPU Acceleration Not Available"
- **Solution**: Install NVIDIA RAPIDS or continue with CPU processing
- **Command**: `pip install cudf-cu12 cuml-cu12 --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple`

#### "Memory Error During Processing"
- **Solution**: Reduce dataset size or increase system RAM
- **Workaround**: Use data sampling for large datasets

#### "Model Training Fails"
- **Check**: Ensure target variable is properly formatted
- **Verify**: All features are numeric (categorical should be encoded)
- **Solution**: Review preprocessing settings

#### "Report Generation Error"
- **Check**: Ensure analysis has been completed
- **Verify**: All required sections have data
- **Solution**: Run complete workflow before generating reports

### Getting Help

1. **Check Logs**: Review `logs/app.log` for detailed error messages
2. **Review Documentation**: Check API documentation for usage details
3. **Sample Data**: Test with provided sample datasets first
4. **Community Support**: Join our Discord community for help

## 🎯 Advanced Features

### GPU Acceleration
- Enable in `config.yaml`: Set `gpu.enabled: true`
- Provides 10-100x speedup for large datasets
- Automatic fallback to CPU if RAPIDS unavailable

### Custom Model Parameters
- Modify `config.yaml` for custom ML settings
- Adjust cross-validation and test split ratios
- Configure specific algorithm parameters

### API Integration
- Add API keys to `.env` file for advanced features
- OpenAI integration for enhanced NLP-to-SQL
- Custom model endpoints for specialized use cases

### Batch Processing
- Process multiple files programmatically
- Automated report generation for regular analysis
- Integration with data pipelines

## 📈 Performance Optimization

### For Large Datasets (1M+ rows)
- Enable GPU acceleration
- Use data sampling for exploratory analysis
- Increase system RAM
- Use efficient file formats (Parquet)

### For Better ML Performance
- Feature engineering and selection
- Hyperparameter tuning
- Ensemble methods
- Cross-validation optimization

### For Faster Processing
- Reduce sample sizes for exploration
- Use specific feature selection
- Optimize preprocessing steps
- Parallel processing where available

## 🔒 Data Privacy & Security

- **Local Processing**: All data processed locally by default
- **No Data Storage**: Data not permanently stored unless explicitly saved
- **API Keys**: Secure storage in environment variables
- **File Permissions**: Appropriate file access controls

## 📞 Support & Community

- **Documentation**: Complete guides and examples
- **Community Forum**: Discord community for questions
- **Bug Reports**: GitHub issues for technical problems
- **Feature Requests**: Submit enhancement suggestions

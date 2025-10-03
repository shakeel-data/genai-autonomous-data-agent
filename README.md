# ⚡GenAI Data Sceince Agent with GPU Acceleration Project | Python, NVIDIA NeMo, RAPIDS cuDF/cuML, TensorFlow, PyTorch, XGBoost, LightGBM, CatBoost, K-Means, SHAP, LIME, Streamlit

<img width="1024" height="466" alt="GenAI-Banner" src="https://github.com/user-attachments/assets/897b5900-df95-4948-96f9-98f4bf1fea62" />
An autonomous data science co-pilot that transforms raw datasets into actionable insights.
Powered by GenAI, SQL, Machine Learning, and NVIDIA GPU acceleration, this agent automates the entire data workflow: cleaning, EDA, modeling, visualization, natural language querying, and dashboard generation.
<p align="center">
  <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" width="40" alt="Python"/>
  <img src="https://developer.download.nvidia.com/compute/cuda/images/NVIDIA-RAPIDS.png" width="100" alt="RAPIDS"/>
  <img src="https://raw.githubusercontent.com/NVIDIA/NeMo/main/docs/_static/nemo_logo.png" width="90" alt="NVIDIA NeMo"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="40" alt="TensorFlow"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="40" alt="PyTorch"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png" width="80" alt="XGBoost"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c9/LightGBM_logo.png" width="80" alt="LightGBM"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/0e/CatBoostLogo.png" width="80" alt="CatBoost"/>
  <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="100" alt="Streamlit"/>
</p>
> Think of it as your Autonomous Data Scientist — always ready, always fast.

⚡ Tech Stack  
Python, NVIDIA NeMo, RAPIDS cuDF/cuML, TensorFlow, PyTorch, XGBoost, LightGBM, CatBoost, K-Means, SHAP, LIME, Streamlit  

<p align="center">
  <!-- Python -->
  <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" width="40" alt="Python"/>
  
  <!-- RAPIDS -->
  <img src="https://raw.githubusercontent.com/rapidsai/docs/main/logo/rapids-logo.png" width="100" alt="RAPIDS"/>
  
  <!-- NVIDIA NeMo -->
  <img src="https://raw.githubusercontent.com/NVIDIA/NeMo/main/docs/_static/nemo_logo.png" width="90" alt="NVIDIA NeMo"/>
  
  <!-- TensorFlow -->
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="40" alt="TensorFlow"/>
  
  <!-- PyTorch -->
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="40" alt="PyTorch"/>
  
  <!-- XGBoost -->
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png" width="80" alt="XGBoost"/>
  
  <!-- LightGBM -->
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/61/LightGBM_logo.png" width="80" alt="LightGBM"/>
  
  <!-- CatBoost -->
  <img src="https://raw.githubusercontent.com/catboost/catboost/master/catboost/logo/catboost_logo_black.png" width="90" alt="CatBoost"/>
  
  <!-- Streamlit -->
  <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="100" alt="Streamlit"/>
</p>

## ✨ Key Features
- ✅ Data Preprocessing – Cleans and preprocesses raw datasets automatically (missing values, scaling, encoding).
- ✅ Exploratory Data Analysis (EDA) – Generates correlations, distributions, and smart visualizations.
- ✅ Automated ML Modeling – Trains & benchmarks multiple models (XGBoost, LightGBM, PyTorch, TensorFlow).
- ✅ GPU Acceleration – Leverages NVIDIA RAPIDS for blazing-fast DataFrame ops and ML pipelines.
- ✅ Explainable AI – Explains predictions in plain English using SHAP & LIME.
- ✅ Natural Language → SQL – Converts plain English queries into optimized SQL with GenAI + NVIDIA NeMo.
- ✅ Interactive Dashboards – Generates insights in Streamlit dashboards exportable to PDF/Excel.
- ✅ Autonomous Agent Loop – Iteratively improves models, queries, and dashboards without manual intervention.

## 🛠️ Tech Stack
- Core: Python, SQL, Streamlit
- ML/DL: TensorFlow, PyTorch, XGBoost, LightGBM, Prophet, K-Means
- GenAI: NVIDIA NeMo, Hugging Face Transformers
- Explainability: SHAP, LIME
- Data Processing: Pandas, RAPIDS cuDF, cuML
- Visualization: Matplotlib, Seaborn, Plotly
- Deployment: Docker + Streamlit + NVIDIA GPU acceleration

## ⚡ Demo
- 🎥 2 Minute Video
<a href="https://github.com/shakeel-data/genai-autonomous-data-agent/blob/main/demo/Demo-video.mp4">Demo</a>

## Getting Started
1. Clone the repo
```terminal
git clone https://github.com/yourusername/genai_autonomous_data_agent.git
cd genai_autonomous_data_agent
```

2. Create virtual environment
```terminal
conda create -n genai_agent python=3.10 -y
conda activate genai_agent
```

3. Install dependencies
```terminal
pip install -r requirements.txt
```
- For GPU acceleration with RAPIDS,
<a href="https://docs.rapids.ai/install">follow RAPIDS install guide</a>

4. Run the Streamlit app
```terminal
streamlit run app.py
```

## 📊 Example Workflow
- Upload raw dataset (.csv, .xlsx).
- Agent automatically cleans and preprocesses.
- Generates EDA reports and key visualizations.
- Benchmarks multiple ML models with GPU acceleration.
- Provides plain-English explanations of predictions.
- Users query with natural language → agent runs SQL under the hood.
- Interactive dashboard with charts & exportable insights.

## 📌 Use Cases
- Business analytics on sales, churn, revenue
- Healthcare predictions (diagnosis, outcomes)
- Finance (fraud detection, forecasting)
- Retail insights (segmentation, demand prediction)
- Education analytics (student performance, outcomes)

## 🔮 Roadmap
 - Autonomous Data-to-Insights Agent Loop
 - Fine-tuned domain-specific NeMo LLM for SQL generation
 - Integration with LangChain for multi-agent orchestration
 - Deploy on NVIDIA Triton Inference Server for scalability
 - Add multi-dataset benchmarking mode
 - Cloud deployment (AWS/GCP/Azure)

## 🏆 NVIDIA GTC 2025 Submission
This project is submitted to the NVIDIA GTC 2025 Golden Ticket Challenge, showcasing how GPU-accelerated, open-source innovation can power next-gen autonomous data science workflows.

## 📚 Platform Capabilities
Advanced Technologies:
- NVIDIA RAPIDS cuDF/cuML → GPU acceleration for 10–100x speedups
- AutoML Pipeline → Train 6+ algorithms automatically
- SHAP & LIME → Real-time model explanations
- NLP-to-SQL → Chat with your data in plain English
- Smart Feature Engineering → Automated data preprocessing
- Multi-format Reports → PDF, HTML, DOCX generation

## 📈 Business Intelligence
- Automated EDA → Statistical analysis & insights
- Predictive Modeling → Classification & regression
- Interactive Visualizations → Plotly-powered charts
- Data Quality Assessment → Missing values & outliers
- Feature Importance → Understand key drivers
- Actionable Recommendations → Business-ready insights

## ⚡ Quick Start Guide (5 Minutes)
Step 1: Data Upload
- Upload CSV/Excel file or use sample datasets
- Automatic data quality assessment
- Smart preprocessing with GPU acceleration

Step 2: Exploratory Analysis
- One-click comprehensive EDA
- Statistical summaries + correlations
- Interactive visualizations

Step 3: ML Training
- AutoML with 6+ algorithms
- Cross-validation & performance metrics
- Feature importance analysis

Step 4: Explainability
- SHAP global explanations
- LIME local explanations
- Business-friendly insights

Step 5: Natural Language SQL
- Ask questions in plain English
- Automatic SQL generation
- Interactive query results

Step 6: Generate Reports
- Comprehensive analysis reports
- Multiple formats: PDF / HTML / DOCX
- Executive summaries

## 🤝 Contributing

Contributions, feedback & ideas are always welcome

Fork the repo, raise issues, or open pull requests to make this project stronger.

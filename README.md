# AI-Powered Data Analysis with GPU Acceleration & Advanced ML

An autonomous data science co-pilot that transforms raw datasets into actionable insights.
Powered by GenAI, SQL, Machine Learning, and NVIDIA GPU acceleration, this agent automates the entire data workflow: cleaning, EDA, modeling, visualization, natural language querying, and dashboard generation.

> 💡 Think of it as your Autonomous Data Scientist — always ready, always fast.

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
- 🎥 Video Demo: [link]
- 📂 Live Project / Streamlit App: [link]

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
For GPU acceleration with RAPIDS, follow RAPIDS install guide [https://docs.rapids.ai/install/]

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

## 🤝 Contributing
Contributions, feedback, and ideas are always welcome. Feel free to fork the repo, raise issues, or open pull requests to make this project stronger.

## 🏆 NVIDIA GTC 2025 Submission
This project is submitted to the NVIDIA GTC 2025 Golden Ticket Challenge, showcasing how GPU-accelerated, open-source innovation can power next-gen autonomous data science workflows.

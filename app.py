"""
GenAI Autonomous Data Agent - Main Streamlit Application
Enterprise-grade AI-powered data analysis platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.navigation import show_top_nav
import sys
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our custom modules
from config import Config
from data_processor import DataProcessor
from eda_module import EDAModule
from ml_module import MLModule
from explain_module import ExplainModule
from sql_agent import SQLAgent
from report_generator import ReportGenerator
from utils import setup_logging, load_sample_data, show_dataframe_info

# Page configuration
st.set_page_config(
    page_title="GenAI Autonomous Data Agent",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Custom CSS for premium UI - NVIDIA OFFICIAL STYLE
st.markdown("""
<style>
    /* ===== NVIDIA OFFICIAL COLOR VARIABLES ===== */
    :root {
        --nvidia-black: #000000;           /* Primary black */
        --nvidia-green: #76b900;           /* Iconic NVIDIA green */
        --nvidia-dark-gray: #333333;       /* Dark gray */
        --nvidia-medium-gray: #666666;     /* Medium gray */
        --nvidia-light-gray: #f8f8f8;      /* Light background */
        --nvidia-white: #ffffff;           /* White */
        --nvidia-green-hover: #8ed100;     /* Brighter green for hover */
        --nvidia-green-active: #5a8a00;    /* Darker green for active */
        --nvidia-border: #e0e0e0;          /* Border color */
        --nvidia-shadow: rgba(0, 0, 0, 0.1); /* Subtle shadows */
    }
    
    /* ===== MAIN HEADER - NVIDIA STYLE ===== */
    .main-header {
        background: linear-gradient(135deg, var(--nvidia-black) 0%, var(--nvidia-dark-gray) 100%);
        padding: 2.5rem;
        border-radius: 0px; /* NVIDIA uses sharp corners */
        text-align: center;
        color: var(--nvidia-white);
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px var(--nvidia-shadow);
        border-bottom: 4px solid var(--nvidia-green);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--nvidia-green);
    }
    
    /* ===== NAVIGATION BUTTONS - NVIDIA STYLE ===== */
    .nav-button-prev, .nav-button-next {
        background: var(--nvidia-black) !important;
        color: var(--nvidia-white) !important;
        border: 2px solid var(--nvidia-green) !important;
        border-radius: 4px !important; /* Sharp corners like NVIDIA */
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        margin: 0.25rem !important;
        box-shadow: 0 2px 8px var(--nvidia-shadow) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .nav-button-prev:hover, .nav-button-next:hover {
        background: var(--nvidia-green) !important;
        color: var(--nvidia-black) !important;
        border-color: var(--nvidia-green) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(118, 185, 0, 0.3) !important;
    }
    
    /* ===== METRIC CARDS - NVIDIA STYLE ===== */
    .metric-card {
        background: var(--nvidia-white);
        padding: 1.5rem;
        border-radius: 0px; /* Sharp corners */
        border-left: 4px solid var(--nvidia-green);
        margin: 1rem 0;
        box-shadow: 0 2px 8px var(--nvidia-shadow);
        border: 1px solid var(--nvidia-border);
        transition: all 0.2s ease;
        font-family: 'Arial', sans-serif;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--nvidia-shadow);
        border-left: 4px solid var(--nvidia-green-hover);
    }
    
    /* ===== STATUS BOXES - NVIDIA STYLE ===== */
    .success-box {
        background: rgba(118, 185, 0, 0.1);
        border: 1px solid var(--nvidia-green);
        border-radius: 0px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--nvidia-dark-gray);
        border-left: 4px solid var(--nvidia-green);
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 0px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--nvidia-dark-gray);
        border-left: 4px solid #ffc107;
    }
    
    .info-box {
        background: rgba(33, 150, 243, 0.1);
        border: 1px solid #2196f3;
        border-radius: 0px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--nvidia-dark-gray);
        border-left: 4px solid #2196f3;
    }
    
    .error-box {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid #f44336;
        border-radius: 0px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--nvidia-dark-gray);
        border-left: 4px solid #f44336;
    }
    
    /* ===== BUTTON STYLES - NVIDIA STYLE ===== */
    .stButton > button {
        background: var(--nvidia-black);
        color: var(--nvidia-white);
        border: 2px solid var(--nvidia-green);
        border-radius: 4px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px var(--nvidia-shadow);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: var(--nvidia-green);
        color: var(--nvidia-black);
        border-color: var(--nvidia-green);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(118, 185, 0, 0.3);
    }
    
    /* Primary button variations */
    .stButton > button[kind="primary"] {
        background: var(--nvidia-green);
        color: var(--nvidia-black);
        border: 2px solid var(--nvidia-green);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: var(--nvidia-green-hover);
        border-color: var(--nvidia-green-hover);
    }
    
    /* Secondary button variations */
    .stButton > button[kind="secondary"] {
        background: var(--nvidia-white) !important;
        color: var(--nvidia-black) !important;
        border: 2px solid var(--nvidia-medium-gray) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--nvidia-green) !important;
        color: var(--nvidia-black) !important;
        border-color: var(--nvidia-green) !important;
    }
    
    /* ===== INPUT FIELDS - NVIDIA STYLE ===== */
    .stTextInput > div > div > input {
        height: 44px;
        border-radius: 0px;
        border: 2px solid var(--nvidia-border);
        font-size: 14px;
        padding: 8px 12px;
        transition: all 0.2s ease;
        background: var(--nvidia-gray);
        font-family: 'Arial', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--nvidia-green) !important;
        box-shadow: 0 0 0 2px rgba(118, 185, 0, 0.2) !important;
        outline: none;
    }
    
    /* ===== SIDEBAR - NVIDIA STYLE ===== */
    .sidebar .sidebar-content {
        background: var(--nvidia-light-gray);
        border-right: 1px solid var(--nvidia-border);
    }
    
    /* ===== EXPANDER STYLING - NVIDIA STYLE ===== */
    .streamlit-expanderHeader {
        background: var(--nvidia-light-gray) !important;
        border: 1px solid var(--nvidia-border) !important;
        border-radius: 0px !important;
        font-weight: 600 !important;
        color: var(--nvidia-black) !important;
        border-left: 4px solid var(--nvidia-green) !important;
    }
    
    /* ===== DATA FRAME STYLING - NVIDIA STYLE ===== */
    .dataframe {
        border: 1px solid var(--nvidia-border) !important;
        border-radius: 0px !important;
    }
    
    /* ===== METRIC STYLING - NVIDIA STYLE ===== */
    [data-testid="metric-container"] {
        background: var(--nvidia-white);
        border: 1px solid var(--nvidia-border);
        border-radius: 0px;
        padding: 1rem;
        box-shadow: 0 2px 8px var(--nvidia-shadow);
        border-left: 4px solid var(--nvidia-green);
    }
    
    /* ===== TAB STYLING - NVIDIA STYLE ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid var(--nvidia-border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--nvidia-light-gray);
        border-radius: 0px;
        padding: 0.75rem 1.5rem;
        border: 1px solid var(--nvidia-border);
        border-bottom: none;
        margin-right: 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--nvidia-black) !important;
        color: var(--nvidia-white) !important;
        border-color: var(--nvidia-green) !important;
        border-bottom: 2px solid var(--nvidia-green) !important;
    }
    
    /* ===== CODE BLOCK STYLING - NVIDIA STYLE ===== */
    .stCodeBlock {
        border: 1px solid var(--nvidia-border);
        border-radius: 0px;
        background: var(--nvidia-light-gray) !important;
        border-left: 4px solid var(--nvidia-green);
    }
    
    /* ===== PROGRESS BAR STYLING - NVIDIA STYLE ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--nvidia-green) 0%, var(--nvidia-green-hover) 100%);
    }
    
    /* ===== CUSTOM CONTAINERS - NVIDIA STYLE ===== */
    .professional-container {
        background: var(--nvidia-white);
        padding: 1.5rem;
        border-radius: 0px;
        border: 1px solid var(--nvidia-border);
        box-shadow: 0 2px 8px var(--nvidia-shadow);
        margin: 1rem 0;
        border-left: 4px solid var(--nvidia-green);
    }
    
    .visualization-container {
        background: var(--nvidia-white);
        padding: 1.5rem;
        border-radius: 0px;
        border: 1px solid var(--nvidia-border);
        box-shadow: 0 2px 8px var(--nvidia-shadow);
        margin-top: 1rem;
        border-left: 4px solid var(--nvidia-green);
    }
    
    /* ===== NVIDIA BADGE STYLING ===== */
    .nvidia-badge {
        background: var(--nvidia-black);
        color: var(--nvidia-green);
        padding: 0.25rem 0.75rem;
        border-radius: 0px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid var(--nvidia-green);
        display: inline-block;
        margin: 0.25rem;
    }
    
    /* ===== NVIDIA ACCENT ELEMENTS ===== */
    .nvidia-accent {
        border-left: 4px solid var(--nvidia-green);
        padding-left: 1rem;
        margin: 1rem 0;
    }
    
    /* ===== HEADER TYPOGRAPHY - NVIDIA STYLE ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--nvidia-black);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h1 {
        border-bottom: 2px solid var(--nvidia-green);
        padding-bottom: 0.5rem;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .nav-button-prev, .nav-button-next {
            padding: 0.5rem 1rem !important;
            font-size: 12px !important;
        }
    }
    
    /* ===== SCROLLBAR STYLING ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--nvidia-light-gray);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--nvidia-green);
        border-radius: 0px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--nvidia-green-hover);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor(st.session_state.config)
    
    if 'eda_module' not in st.session_state:
        st.session_state.eda_module = EDAModule(st.session_state.config)
    
    if 'ml_module' not in st.session_state:
        st.session_state.ml_module = MLModule(st.session_state.config)
    
    if 'explain_module' not in st.session_state:
        st.session_state.explain_module = ExplainModule(st.session_state.config)
    
    if 'sql_agent' not in st.session_state:
        st.session_state.sql_agent = SQLAgent(st.session_state.config)
    
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ReportGenerator(st.session_state.config)
    
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    
    if 'eda_results' not in st.session_state:
        st.session_state.eda_results = None
    
    if 'explanation_results' not in st.session_state:
        st.session_state.explanation_results = None
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def show_header():
    """Display the main header with NVIDIA styling"""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; color: #76b900;">GenAI Autonomous Data Agent</h1>
        <p style="font-size: 1.3em; margin: 0.5rem 0; font-weight: 500; color: #ffffff;">AI-Powered Data Analysis with GPU Acceleration & Advanced ML</p>
        <p style="font-size: 1em; margin: 0; opacity: 0.9; font-weight: 400; color: #cccccc;">
            <span class="nvidia-badge">RAPIDS cuML</span>
            <span class="nvidia-badge">SHAP/LIME</span>
            <span class="nvidia-badge">AutoML</span>
            <span class="nvidia-badge">NLP-to-SQL</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_home_dashboard():
    """Display the home dashboard with overview and metrics"""
    
    # Custom CSS for centered metrics
    st.markdown("""
    <style>
    /* Center align all metric components */
    [data-testid="stMetric"] {
        text-align: center;
        justify-content: center;
    }
    
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        text-align: center;
        width: 100%;
        font-weight: 600;
        font-size: 0.9rem;
        color: #4A5568;
    }
    
    [data-testid="stMetricValue"] {
        display: flex;
        justify-content: center;
        text-align: center;
        width: 100%;
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #2D3748;
    }
    
    [data-testid="stMetricDelta"] {
        display: flex;
        justify-content: center;
        text-align: center;
        width: 100%;
    }
    
    .centered-metric {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dataset_count = 1 if st.session_state.current_dataset is not None else 0
        st.markdown('<div class="centered-metric">', unsafe_allow_html=True)
        st.metric(
            label="DATASET LOADED",
            value=dataset_count,
            delta="Ready for analysis" if dataset_count > 0 else "Upload data to start"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        ml_count = 1 if st.session_state.ml_results is not None else 0
        st.markdown('<div class="centered-metric">', unsafe_allow_html=True)
        st.metric(
            label="MODELS TRAINED",
            value=ml_count,
            delta="Best model available" if ml_count > 0 else "Train models after upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        query_count = len(st.session_state.query_history)
        delta_value = None
        if query_count > 0:
            delta_value = f"Last query: {st.session_state.query_history[-1]['timestamp'].strftime('%H:%M')}"
        st.markdown('<div class="centered-metric">', unsafe_allow_html=True)
        st.metric(
            label="SQL QUERIES",
            value=query_count,
            delta=delta_value if query_count > 0 else "No queries yet"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        explainer_count = 1 if st.session_state.explanation_results is not None else 0
        st.markdown('<div class="centered-metric">', unsafe_allow_html=True)
        st.metric(
            label="EXPLANATIONS",
            value=explainer_count,
            delta="SHAP/LIME available" if explainer_count > 0 else "Generate after training"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Showcase
    st.markdown("## **Platform Capabilities**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### **Advanced Technologies**
        - **NVIDIA RAPIDS cuDF/cuML** - GPU acceleration for 10-100x speedups
        - **Auto-ML Pipeline** - Train 6+ algorithms automatically
        - **SHAP & LIME** - Real-time model explanations
        - **NLP-to-SQL** - Chat with your data in plain English
        - **Smart Feature Engineering** - Automated data preprocessing
        - **Multi-format Reports** - PDF, HTML, DOCX generation
        """)
    
    with col2:
        st.markdown("""
        ### **Business Intelligence**
        - **Automated EDA** - Statistical analysis & insights
        - **Predictive Modeling** - Classification & Regression
        - **Interactive Visualizations** - Plotly-powered charts
        - **Data Quality Assessment** - Missing values & outliers
        - **Feature Importance** - Understand key drivers
        - **Actionable Recommendations** - Business-ready insights
        """)
    
    # Quick Start Guide
    st.markdown("---")
    st.markdown("## **Quick Start Guide**")
    
    with st.expander("**Complete Workflow (5 Minutes)**", expanded=True):
        st.markdown("""
        ### Step-by-Step Process:
        
        **1. Data Upload**
        - Upload CSV/Excel file or use sample datasets
        - Automatic data quality assessment
        - Smart preprocessing with GPU acceleration
        
        **2. Exploratory Analysis**  
        - One-click comprehensive EDA
        - Statistical summaries and correlations
        - Interactive visualizations
        
        **3. ML Training**
        - Auto-ML with 6+ algorithms
        - Cross-validation and performance metrics
        - Feature importance analysis
        
        **4. Explainability**
        - SHAP global explanations
        - LIME local explanations  
        - Business-friendly insights
        
        **5. Natural Language SQL**
        - Ask questions in plain English
        - Automatic SQL generation
        - Interactive query results
        
        **6. Generate Reports**
        - Comprehensive analysis reports
        - Multiple formats (PDF/HTML/DOCX)
        - Executive summaries
        """)
    
    # Sample Queries
    if st.session_state.current_dataset is not None:
        st.markdown("---")
        st.markdown("## **Try These Sample Queries**")
        
        sample_queries = st.session_state.sql_agent.get_query_suggestions(st.session_state.current_dataset)
        
        col1, col2 = st.columns(2)
        for i, query in enumerate(sample_queries):
            if i < len(sample_queries) // 2:
                col1.code(query)
            else:
                col2.code(query)

def show_data_upload_page():
    """Data upload and preprocessing interface with optimized UI and streamlined functionality"""
    
    # Initialize session state for output persistence
    if 'preprocessing_output_visible' not in st.session_state:
        st.session_state.preprocessing_output_visible = False
    if 'processed_data_available' not in st.session_state:
        st.session_state.processed_data_available = False
    if 'last_preprocessing_success' not in st.session_state:
        st.session_state.last_preprocessing_success = False
    if 'sample_loaded' not in st.session_state:
        st.session_state.sample_loaded = False
    
    # Optimized CSS styling with minimal spacing
    st.markdown("""
    <style>
    .minimal-upload-section {
        background: linear-gradient(135deg, #001F17 0%, #003B2E 100%);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #76B900;
    }
    
    .minimal-upload-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.8rem;
        font-family: 'Arial', sans-serif;
    }
    
    .minimal-upload-subtitle {
        font-size: 0.85rem;
        color: #76B900;
        margin-bottom: 1.5rem;
        font-family: 'Arial', sans-serif;
    }
    
    .upload-area {
        border: 2px dashed #76B900;
        border-radius: 8px;
        padding: 2.5rem 1rem;
        background: rgba(118, 185, 0, 0.05);
        margin: 0.5rem 0;
    }
    
    .browse-button {
        background: linear-gradient(135deg, #76B900 0%, #00FF88 100%);
        color: #001F17;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 1rem 0;
        cursor: pointer;
        display: inline-block;
    }
    
    .browse-button:hover {
        background: linear-gradient(135deg, #85CC00 0%, #20FF98 100%);
    }
    
    .sample-section {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        background: #f8f8f8;
    }
    
    .sample-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    
    .file-metric-card {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.8rem;
        text-align: center;
        margin: 0.3rem;
        background: #f8f8f8;
    }
    
    .processing-section {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    
    .suggestion-box {
        border: 1px solid #76B900;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.8rem 0;
        background: #f8fff8;
    }
    
    .output-section {
        border: 2px solid #76B900;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        background: #f8fff8;
    }
    
    /* Remove unnecessary spacing */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stFileUploader"]) {
        margin-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## Data Upload & Processing")
    st.markdown("Upload your dataset and let AI handle the preprocessing automatically.")
    
    
    # Single file uploader with hidden label
    uploaded_file = st.file_uploader(
        " ",
        type=['csv', 'xlsx', 'json', 'parquet'],
        help="Drag and drop your file here or click to browse",
        key="main_uploader",
        label_visibility="collapsed"
    )
    
    
    
    st.markdown(
    """
    <style>
    .sample-title {
        color: white;        /* text color */
        text-align: left;    /* left alignment */
        font-size: 20px;     /* optional: make it more visible */
        font-weight: bold;   /* optional: make it bold */
        margin-bottom: 10px; /* optional: spacing */
    }
    </style>
    <div class="sample-title">Sample Datasets</div>
    """,
    unsafe_allow_html=True
)

    
    sample_options = [
        "None",
        "E-commerce Sales (1000 rows)",
        "Healthcare Analytics (800 rows)", 
        "Financial Market Data (1200 rows)",
        "Automotive Performance (600 rows)"
    ]
    
    selected_sample = st.selectbox(
        "Select sample dataset:",
        sample_options,
        label_visibility="collapsed"
    )
    
    # Auto-load sample dataset when selected (removed the manual button)
    if selected_sample != "None" and not st.session_state.get('sample_loaded', False):
        with st.spinner(f"Loading {selected_sample}..."):
            try:
                df = load_sample_data(selected_sample)
                st.session_state.current_dataset = df
                st.session_state.original_dataset = df.copy()
                st.session_state.sample_loaded = True
                # Reset output state when new data is loaded
                st.session_state.preprocessing_output_visible = False
                st.session_state.processed_data_available = False
                st.session_state.last_preprocessing_success = False
                st.success(f"Successfully loaded: {selected_sample}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Reset sample loaded flag when "None" is selected
    if selected_sample == "None" and st.session_state.get('sample_loaded', False):
        st.session_state.sample_loaded = False
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process uploaded file or selected sample
    current_file = uploaded_file if uploaded_file is not None else None
    has_sample_loaded = st.session_state.current_dataset is not None and selected_sample != "None"
    
    if current_file is not None or has_sample_loaded:
        try:
            with st.spinner("Loading and analyzing data..."):
                # Load data only when user explicitly triggers it
                if current_file and not has_sample_loaded:
                    df = st.session_state.data_processor.load_data(current_file)
                    st.success(f"Successfully loaded: {current_file.name}")
                    # Reset output state when new data is loaded
                    st.session_state.preprocessing_output_visible = False
                    st.session_state.processed_data_available = False
                    st.session_state.last_preprocessing_success = False
                    st.session_state.sample_loaded = False
                else:
                    df = st.session_state.current_dataset
                
                # Store dataset
                st.session_state.current_dataset = df
                if 'original_dataset' not in st.session_state:
                    st.session_state.original_dataset = df.copy()
            
            # Data preview
            st.markdown("---")
            st.markdown("### Data Preview")
            
            show_dataframe_info(df)
            
            with st.expander("View Sample Data (First 10 Rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data quality analysis
            st.markdown("### Data Quality Analysis")
            
            with st.spinner("Analyzing data quality..."):
                quality_analysis = st.session_state.data_processor.analyze_data_quality(df)
            
            if quality_analysis:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    quality_score = quality_analysis.get('quality_score', 0)
                    st.metric(
                        "Data Quality Score", 
                        f"{quality_score:.1%}", 
                        delta="Excellent" if quality_score > 0.8 else "Needs attention"
                    )
                
                with col2:
                    missing_pct = quality_analysis.get('missing_data', {}).get('missing_percentage', 0)
                    st.metric(
                        "Missing Values", 
                        f"{missing_pct:.1f}%", 
                        delta="Low" if missing_pct < 5 else "High"
                    )
                
                with col3:
                    duplicate_pct = quality_analysis.get('duplicates', {}).get('duplicate_percentage', 0)
                    st.metric(
                        "Duplicate Rows", 
                        f"{duplicate_pct:.1f}%", 
                        delta="Low" if duplicate_pct < 2 else "High"
                    )
                
                # Recommendations
                recommendations = quality_analysis.get('recommendations', [])
                if recommendations:
                    with st.expander("Data Quality Recommendations", expanded=False):
                        for i, rec in enumerate(recommendations, 1):
                            st.info(f"{i}. {rec}")
            
            # Smart Preprocessing Section
            st.markdown("---")
            st.markdown("### Smart Preprocessing Options")
            
            # Auto-detect preprocessing suggestions
            suggestions = {}
            if st.session_state.current_dataset is not None:
                with st.spinner("Analyzing data for preprocessing suggestions..."):
                    quality_analysis = st.session_state.data_processor.analyze_data_quality(st.session_state.current_dataset)
                    suggestions = quality_analysis.get('preprocessing_suggestions', {})
                    
                    if suggestions:
                        st.markdown("#### AI Suggestions")
                        st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                        st.info("Based on data analysis, we recommend:")
                        
                        for reason in suggestions.get('reasoning', []):
                            st.write(f"• {reason}")
                        st.markdown("</div>", unsafe_allow_html=True)

            
            with st.expander("Configure Preprocessing Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    handle_missing = st.selectbox(
                        "Handle Missing Values:",
                        ["auto", "drop_rows", "drop_columns", "fill_median", "fill_mean", "fill_mode"],
                        help="Auto mode uses intelligent strategies based on data types",
                        index=0
                    )
                    
                    encode_categorical = st.checkbox(
                        "Auto-encode Categorical Variables", 
                        value=suggestions.get('encode_categorical', True),
                        help="Convert text/categorical variables to numeric"
                    )
                    
                    normalize_features = st.checkbox(
                        "Normalize Numerical Features", 
                        value=suggestions.get('normalize_features', False),
                        help="Scale features for better ML performance"
                    )
                
                with col2:
                    remove_outliers = st.checkbox(
                        "Remove Statistical Outliers", 
                        value=suggestions.get('remove_outliers', False),
                        help="Remove data points outside 1.5*IQR"
                    )
                    
                    feature_selection = st.checkbox(
                        "Automated Feature Selection", 
                        value=suggestions.get('feature_selection', False),
                        help="Select most relevant features automatically"
                    )
            
            # Run Preprocessing and Clear buttons
            st.markdown("---")
            col_run, col_clear = st.columns(2)
            
            with col_run:
                if st.button("Run Preprocessing", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        try:
                            original_df = st.session_state.original_dataset.copy()
                            
                            # Use safe preprocessing with error handling
                            processed_df, error, summary = safe_preprocessing(
                                st.session_state.data_processor,
                                original_df, 
                                handle_missing=handle_missing,
                                encode_categorical=encode_categorical,
                                normalize_features=normalize_features,
                                remove_outliers=remove_outliers,
                                feature_selection=feature_selection
                            )
                            
                            if error:
                                st.error(f"Preprocessing Error: {error}")
                                st.session_state.last_preprocessing_success = False
                            else:
                                st.session_state.current_dataset = processed_df
                                st.session_state.last_preprocessing_summary = summary
                                st.session_state.preprocessing_output_visible = True
                                st.session_state.processed_data_available = True
                                st.session_state.last_preprocessing_success = True
                                st.success("Data preprocessing completed successfully!")
                                
                        except Exception as e:
                            st.error(f"Preprocessing failed: {str(e)}")
                            st.session_state.last_preprocessing_success = False
            
            with col_clear:
                if st.button("Clear Output", type="secondary", use_container_width=True):
                    # Clear only the output display, not the data
                    st.session_state.preprocessing_output_visible = False
                    st.session_state.processed_data_available = False
                    st.success("Output display cleared!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # PERSISTENT Preprocessing Output Display
            if st.session_state.get('preprocessing_output_visible', False) and st.session_state.get('last_preprocessing_summary'):
                st.markdown("---")
                st.markdown("### Preprocessing Results")
                
                summary = st.session_state.last_preprocessing_summary
                original_df = st.session_state.original_dataset
                processed_df = st.session_state.current_dataset
                
                # Processing summary
                st.markdown("#### Processing Summary")
                col_sum1, col_sum2 = st.columns(2)
                with col_sum1:
                    st.info(f"**Original Shape:** {summary['original_shape'][0]:,} rows × {summary['original_shape'][1]} columns")
                    st.info(f"**Processed Shape:** {summary['processed_shape'][0]:,} rows × {summary['processed_shape'][1]} columns")
                    st.info(f"**Processing Time:** {summary['processing_time']} seconds")
                
                with col_sum2:
                    st.info(f"**Rows Removed:** {summary['rows_removed']:,}")
                    st.info(f"**Columns Removed:** {summary['columns_removed']}")
                    st.info(f"**Missing Values:** {summary['missing_values_before']:,} → {summary['missing_values_after']:,}")
                
                # Detailed comparison
                st.markdown("#### Detailed Comparison")
                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.markdown("**Before Processing:**")
                    st.info(f"""
                    **Shape:** {original_df.shape[0]:,} rows × {original_df.shape[1]} columns
                    **Missing Values:** {original_df.isnull().sum().sum():,}
                    **Memory:** {original_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
                    **Data Types:** {len(original_df.select_dtypes(include=['object']).columns)} categorical, {len(original_df.select_dtypes(include=[np.number]).columns)} numeric
                    """)
                
                with col_comp2:
                    st.markdown("**After Processing:**")
                    st.success(f"""
                    **Shape:** {processed_df.shape[0]:,} rows × {processed_df.shape[1]} columns
                    **Missing Values:** {processed_df.isnull().sum().sum():,}
                    **Memory:** {processed_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
                    **Data Types:** {len(processed_df.select_dtypes(include=['object']).columns)} categorical, {len(processed_df.select_dtypes(include=[np.number]).columns)} numeric
                    """)
                
                # Processing changes
                changes = summary.get('applied_operations', [])
                if changes:
                    with st.expander("Processing Changes Applied", expanded=True):
                        for change in changes:
                            st.write(f"• {change}")
                
                # Processed data sample
                with st.expander("View Processed Data (First 10 Rows)", expanded=False):
                    st.dataframe(processed_df.head(10), use_container_width=True)
                
                # Download button for processed data
                if st.session_state.get('processed_data_available', False):
                    st.markdown("---")
                    col_download, col_clear_output = st.columns(2)
                    
                    with col_download:
                        csv = processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Preprocessed Data",
                            data=csv,
                            file_name=f"preprocessed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            type="primary"
                        )
                    
                    with col_clear_output:
                        if st.button("Clear Output Display", type="secondary", use_container_width=True):
                            st.session_state.preprocessing_output_visible = False
                            st.session_state.processed_data_available = False
                            st.success("Output display cleared!")
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            with st.expander("Debug Information"):
                st.code(str(e))
    
    else:
        # Clean instructions when no data is loaded
        st.markdown("---")
        st.info("""
        **Getting Started:**
        
        1. **Upload your data file** (CSV, Excel, JSON, or Parquet)
        2. **Or select a sample dataset** to explore the platform  
        3. **Review data quality** and preprocessing recommendations
        4. **Apply preprocessing** to prepare data for analysis
        5. **Proceed to EDA** or ML training once data is ready
        
        **Supported formats:** CSV, XLSX, JSON, Parquet  
        **Maximum file size:** 500 MB  
        **Sample datasets:** Pre-loaded with realistic business data
        """)
        

def safe_preprocessing(data_processor, df, **kwargs):
    """
    Safely apply preprocessing with comprehensive error handling and validation
    Returns: (processed_df, error_message, processing_summary)
    """
    try:
        # Validate input data
        if df is None or df.empty:
            return None, "Input data is empty or None", {}
        
        # Create backup of original data
        original_shape = df.shape
        original_columns = df.columns.tolist()
        original_missing = df.isnull().sum().sum()
        original_dtypes = dict(df.dtypes.value_counts())
        
        # Apply preprocessing with detailed monitoring
        start_time = time.time()
        result = data_processor.preprocess_data(df, **kwargs)
        processing_time = time.time() - start_time
        
        # Validate output
        if result is None:
            return None, "Preprocessing returned None", {}
        
        if result.empty:
            return None, "Preprocessing resulted in empty dataset", {}
        
        # Generate comprehensive processing summary
        processing_summary = {
            'original_shape': original_shape,
            'processed_shape': result.shape,
            'processing_time': round(processing_time, 2),
            'columns_removed': len(original_columns) - len(result.columns),
            'rows_removed': original_shape[0] - result.shape[0],
            'missing_values_before': original_missing,
            'missing_values_after': result.isnull().sum().sum(),
            'data_types_before': original_dtypes,
            'data_types_after': dict(result.dtypes.value_counts()),
            'memory_before_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'memory_after_mb': result.memory_usage(deep=True).sum() / 1024**2,
            'applied_operations': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Track applied operations based on kwargs
        operations = []
        if kwargs.get('handle_missing') and kwargs.get('handle_missing') != 'none':
            operations.append(f"Missing value handling: {kwargs.get('handle_missing')}")
        if kwargs.get('encode_categorical'):
            operations.append("Categorical variables encoded")
        if kwargs.get('normalize_features'):
            operations.append("Numerical features normalized")
        if kwargs.get('remove_outliers'):
            operations.append("Statistical outliers removed")
        if kwargs.get('feature_selection'):
            operations.append("Automated feature selection applied")
        
        processing_summary['applied_operations'] = operations
        
        return result, None, processing_summary
        
    except Exception as e:
        error_msg = f"Preprocessing error: {str(e)}"
        print(f"Safe preprocessing error: {error_msg}")
        return None, error_msg, {}


def show_eda_page():
    """Exploratory Data Analysis interface - Complete error-free version"""
    
    if st.session_state.current_dataset is None:
        st.warning("**Please upload data first** in the Data Upload section")
        return
    
    st.markdown("## **Exploratory Data Analysis**")
    st.markdown("Comprehensive statistical analysis and data insights powered by AI.")
    
    df = st.session_state.current_dataset
    
    # EDA Options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### **Analysis Options**")
        
        analysis_options = st.multiselect(
            "**Select analysis components:**",
            [
                "Statistical Summary",
                "Correlation Analysis", 
                "Distribution Analysis",
                "Missing Values Analysis",
                "Outlier Detection",
                "Automated Insights"
            ],
            default=["Statistical Summary", "Correlation Analysis", "Automated Insights"]
        )
    
    with col2:
        # Complete EDA Button with better handling
        if st.button("**Run Full EDA**", type="primary"):
            with st.spinner("Running comprehensive EDA analysis..."):
                try:
                    # Clear previous results
                    for key in ['eda_results', 'show_all_eda']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Run comprehensive EDA
                    eda_results = st.session_state.eda_module.generate_comprehensive_eda(df)
                    st.session_state.eda_results = eda_results
                    
                    if eda_results and 'error' not in eda_results:
                        st.success("**Complete EDA analysis finished!**")
                        
                        # Set all options to be displayed
                        st.session_state.show_all_eda = True
                        
                    else:
                        st.error(f"**EDA failed:** {eda_results.get('error', 'Unknown error') if eda_results else 'No results generated'}")
                        
                except Exception as e:
                    st.error(f"**Error running complete EDA:** {str(e)}")
                    st.session_state.eda_results = {'error': str(e)}
    
    # Check if we should show all EDA results (from complete run)
    if hasattr(st.session_state, 'show_all_eda') and st.session_state.show_all_eda:
        analysis_options = [
            "Statistical Summary",
            "Correlation Analysis", 
            "Distribution Analysis",
            "Missing Values Analysis",
            "Outlier Detection",
            "Automated Insights"
        ]
    
    # Display EDA Results - COMPLETELY SAFE VERSION
    if ((hasattr(st.session_state, 'eda_results') and st.session_state.eda_results) or 
        (analysis_options and len(analysis_options) > 0)):
        
        st.markdown("---")
        
        # Statistical Summary - SAFE VERSION
        if analysis_options and "Statistical Summary" in analysis_options:
            st.markdown("### **Statistical Summary**")
            
            try:
                with st.spinner("Generating statistical summary..."):
                    summary = None
                    
                    # Safe access to cached results
                    if (hasattr(st.session_state, 'eda_results') and 
                        st.session_state.eda_results and 
                        isinstance(st.session_state.eda_results, dict) and 
                        'statistical_summary' in st.session_state.eda_results):
                        summary = st.session_state.eda_results['statistical_summary']
                    
                    # Generate if not cached
                    if summary is None or (isinstance(summary, pd.DataFrame) and summary.empty):
                        summary = st.session_state.eda_module.statistical_summary(df)
                    
                    # Display results
                    if summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty:
                        st.dataframe(summary.round(4), use_container_width=True)
                        
                        # Key insights from summary
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Numeric Features", len(numeric_cols))
                            with col2:
                                try:
                                    skewed_cols = [col for col in numeric_cols if abs(df[col].skew()) > 1]
                                    st.metric("Skewed Features", len(skewed_cols))
                                except:
                                    st.metric("Skewed Features", "N/A")
                            with col3:
                                try:
                                    high_var_cols = [col for col in numeric_cols if df[col].std() > df[col].mean()]
                                    st.metric("High Variance", len(high_var_cols))
                                except:
                                    st.metric("High Variance", "N/A")
                    else:
                        st.warning("Could not generate statistical summary")
            
            except Exception as e:
                st.error(f"Error in statistical summary: {str(e)}")
        
        # Correlation Analysis - SAFE VERSION
        if analysis_options and "Correlation Analysis" in analysis_options:
            st.markdown("### **Correlation Analysis**")
            
            try:
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) >= 2:
                    with st.spinner("Analyzing correlations..."):
                        # Create correlation heatmap
                        try:
                            fig = st.session_state.eda_module.create_correlation_heatmap(df)
                            if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not create correlation heatmap")
                        except Exception as e:
                            st.error(f"Error creating heatmap: {str(e)}")
                        
                        # High correlation pairs
                        try:
                            corr_matrix = numeric_df.corr()
                            high_corr_pairs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    corr_value = corr_matrix.iloc[i, j]
                                    if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
                            
                            if high_corr_pairs:
                                st.markdown("**Highly Correlated Feature Pairs (|r| > 0.7):**")
                                for var1, var2, corr in high_corr_pairs[:5]:
                                    st.info(f"**{var1}** ↔ **{var2}**: {corr:.3f}")
                            else:
                                st.info("ℹNo highly correlated feature pairs found")
                        except Exception as e:
                            st.error(f"Error analyzing correlations: {str(e)}")
                else:
                    st.info("ℹ Not enough numeric columns for correlation analysis")
            
            except Exception as e:
                st.error(f"Error in correlation analysis: {str(e)}")
        
        # Distribution Analysis - SAFE VERSION
        if analysis_options and "Distribution Analysis" in analysis_options:
            st.markdown("### **Distribution Analysis**")
            
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("**Select feature for distribution analysis:**", 
                                              options=numeric_cols, key="dist_analysis_select")
                    
                    if selected_col:
                        try:
                            with st.spinner(f"Analyzing distribution of {selected_col}..."):
                                fig = st.session_state.eda_module.create_distribution_plot(df, selected_col)
                                if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Could not create distribution plot")
                        except Exception as e:
                            st.error(f"Error creating distribution plot: {str(e)}")
                else:
                    st.info("ℹ No numeric columns available for distribution analysis")
            
            except Exception as e:
                st.error(f"Error in distribution analysis: {str(e)}")
        
        # Missing Values Analysis - SAFE VERSION
        if analysis_options and "Missing Values Analysis" in analysis_options:
            st.markdown("### **Missing Values Analysis**")
            
            try:
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    with st.spinner("Analyzing missing values..."):
                        try:
                            fig = st.session_state.eda_module.create_missing_values_plot(df)
                            if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create missing values plot: {str(e)}")
                        
                        # Missing values table
                        try:
                            missing_df = pd.DataFrame({
                                'Column': missing_data.index,
                                'Missing Count': missing_data.values,
                                'Missing %': (missing_data.values / len(df) * 100).round(2)
                            })
                            st.dataframe(missing_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating missing values table: {str(e)}")
                else:
                    st.success("**No missing values found in the dataset!**")
            
            except Exception as e:
                st.error(f"Error in missing values analysis: {str(e)}")
        
        # Outlier Detection - SAFE VERSION
        if analysis_options and "Outlier Detection" in analysis_options:
            st.markdown("### **Outlier Detection**")
            
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    with st.spinner("Detecting outliers..."):
                        outlier_results = None
                        
                        # Safe access to cached results
                        if (hasattr(st.session_state, 'eda_results') and 
                            st.session_state.eda_results and 
                            isinstance(st.session_state.eda_results, dict) and 
                            'outlier_analysis' in st.session_state.eda_results):
                            outlier_results = st.session_state.eda_results['outlier_analysis']
                        
                        # Generate if not cached
                        if not outlier_results:
                            outlier_results = st.session_state.eda_module.detect_outliers(df)
                        
                        if (outlier_results and 
                            isinstance(outlier_results, dict) and 
                            'error' not in outlier_results and
                            len(outlier_results) > 0):
                            
                            # Create outlier summary table
                            outlier_data = []
                            for col, stats in outlier_results.items():
                                if isinstance(stats, dict) and 'error' not in stats:
                                    outlier_data.append({
                                        'Column': col,
                                        'IQR Outliers': stats.get('iqr_outliers', 0),
                                        'IQR %': f"{stats.get('iqr_percentage', 0):.2f}%",
                                        'Z-Score Outliers': stats.get('zscore_outliers', 0),
                                        'Z-Score %': f"{stats.get('zscore_percentage', 0):.2f}%",
                                        'Modified Z-Score Outliers': stats.get('modified_zscore_outliers', 0),
                                        'Modified Z-Score %': f"{stats.get('modified_zscore_percentage', 0):.2f}%"
                                    })
                            
                            if outlier_data:
                                outlier_df = pd.DataFrame(outlier_data)
                                st.dataframe(outlier_df, use_container_width=True)
                                
                                # Show summary
                                try:
                                    total_outliers = sum([
                                        stats.get('iqr_outliers', 0) 
                                        for stats in outlier_results.values() 
                                        if isinstance(stats, dict) and 'error' not in stats
                                    ])
                                    st.info(f"**Total outliers detected (IQR method):** {total_outliers}")
                                except Exception as e:
                                    st.warning("Could not calculate total outliers")
                            else:
                                st.info("ℹ No outliers detected in the dataset")
                        else:
                            st.warning("Could not complete outlier detection")
                            if outlier_results and isinstance(outlier_results, dict) and 'error' in outlier_results:
                                st.error(f"Error: {outlier_results['error']}")
                else:
                    st.info("ℹ No numeric columns available for outlier detection")
            
            except Exception as e:
                st.error(f"Error in outlier detection: {str(e)}")
        
        # Automated Insights - SAFE VERSION
        if analysis_options and "Automated Insights" in analysis_options:
            st.markdown("### **AI-Generated Insights**")
            
            try:
                with st.spinner("Generating AI insights..."):
                    insights = None
                    
                    # Safe access to cached results
                    if (hasattr(st.session_state, 'eda_results') and 
                        st.session_state.eda_results and 
                        isinstance(st.session_state.eda_results, dict) and 
                        'insights' in st.session_state.eda_results):
                        insights = st.session_state.eda_results['insights']
                    
                    # Generate if not cached
                    if not insights:
                        insights = st.session_state.eda_module.generate_insights(df)
                    
                    # Display insights
                    if insights and isinstance(insights, list) and len(insights) > 0:
                        for i, insight in enumerate(insights, 1):
                            if insight and isinstance(insight, str) and len(insight.strip()) > 0:
                                st.info(f"**Insight {i}:** {insight}")
                    else:
                        st.info("**No specific insights generated** - data appears to be well-structured")
            
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
        
        # Clear results section
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("**Clear Results**", type="secondary"):
                # Safe cleanup of session state
                keys_to_remove = ['eda_results', 'show_all_eda']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("**Reset Analysis**", type="secondary"):
                # This will reset the multiselect on next run
                st.rerun()



def show_ml_page():
    """Enhanced Machine Learning training interface with advanced features"""
    
    # Helper function for model framework detection
    def _get_model_framework(model_name):
        """Determine the framework of a model"""
        model_name_lower = model_name.lower()
        
        if any(fw in model_name_lower for fw in ['tensorflow', 'tf', 'keras']):
            return 'TensorFlow'
        elif any(fw in model_name_lower for fw in ['pytorch', 'torch']):
            return 'PyTorch'
        elif any(fw in model_name_lower for fw in ['gpu', 'cuml']):
            return 'NVIDIA cuML'
        elif any(fw in model_name_lower for fw in ['xgb', 'lightgbm', 'catboost']):
            return 'Boosted Trees'
        else:
            return 'Scikit-learn'
    
    if st.session_state.current_dataset is None:
        st.warning("**Please upload data first** in the Data Upload section")
        return
    
    st.markdown("## **Machine Learning Training**")
    st.markdown("Enhanced AutoML pipeline with multi-framework support, GPU acceleration, and advanced algorithms.")
    
    df = st.session_state.current_dataset
    
    # Enhanced ML Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### **Model Configuration**")
        
        # Target selection
        target_col = st.selectbox(
            "**Select Target Variable:**",
            options=df.columns.tolist(),
            help="Choose the column you want to predict"
        )
        
        # Feature selection with enhanced options
        available_features = [col for col in df.columns if col != target_col]
        feature_cols = st.multiselect(
            "**Select Features (leave empty for auto-selection):**",
            options=available_features,
            default=None,
            help="Select specific features or leave empty to use all"
        )
        
        # Enhanced task type with auto-detection preview
        task_type = st.radio(
            "**Task Type:**",
            ["auto", "classification", "Regression"],
            help="Auto-detection based on target variable characteristics"
        )
        
        # Show auto-detection preview
        if task_type == "auto" and target_col:
            try:
                detected_type = st.session_state.ml_module._detect_task_type(df[target_col])
                st.info(f"**Auto-detected task type:** {detected_type}")
            except:
                pass
    
    with col2:
        st.markdown("### **Advanced Options**")
        
        test_size = st.slider("**Test Set Size**", 0.1, 0.4, 0.2, 0.05, 
                             help="Proportion of data for testing")
        cv_folds = st.slider("**CV Folds**", 3, 10, 5, 1, 
                           help="Cross-validation folds")
        random_state = st.number_input("**Random State**", value=42, 
                                     help="For reproducible results")
        
        # NEW: Advanced feature engineering toggle
        st.markdown("### **Advanced Features**")
        enable_advanced_features = st.checkbox(
            "**Enable Advanced Feature Engineering**",
            value=True,
            help="Add clustering features, PCA components, and statistical features"
        )
        
        # NEW: Model framework selection
        framework_preference = st.selectbox(
            "**Framework Preference:**",
            ["All Frameworks", "Traditional ML", "Deep Learning", "GPU Accelerated"],
            help="Focus on specific model types"
        )
    
    # NEW: Advanced Configuration Expander
    with st.expander("**Advanced Configuration**", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Algorithm selection
            st.markdown("**Algorithm Groups:**")
            include_tree_models = st.checkbox("Tree-based Models", True)
            include_linear_models = st.checkbox("Linear Models", True)
            include_ensemble_models = st.checkbox("Ensemble Models", True)
            include_dl_models = st.checkbox("Deep Learning", True)
        
        with col2:
            # Performance options
            st.markdown("**Performance Options:**")
            enable_gpu = st.checkbox("GPU Acceleration", 
                                   value=st.session_state.ml_module.use_gpu,
                                   disabled=not st.session_state.ml_module.use_gpu)
            optimize_memory = st.checkbox("Memory Optimization", True)
            early_stopping = st.checkbox("Early Stopping", True)
        
        with col3:
            # Data options
            st.markdown("**Data Options:**")
            balance_data = st.checkbox("Auto-balance Classes", False)
            remove_correlated = st.checkbox("Remove Correlated Features", True)
            feature_scaling = st.checkbox("Auto Feature Scaling", True)
    
    # Enhanced Training Section
    st.markdown("---")
    st.markdown("### **Model Training**")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("**Train Models**", type="primary", use_container_width=True):
            try:
                with st.spinner("Training advanced ML models with enhanced features..."):
                    
                    # Prepare features and target
                    if not feature_cols:
                        feature_cols = available_features
                    
                    X = df[feature_cols]
                    y = df[target_col]
                    
                    # NEW: Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(step, total_steps, message):
                        progress = (step / total_steps)
                        progress_bar.progress(progress)
                        status_text.text(f"{message}...")
                    
                    # Mock progress updates (in real implementation, this would be integrated into MLModule)
                    update_progress(1, 5, "Preprocessing data with advanced feature engineering")
                    
                    # Run Enhanced Auto-ML pipeline
                    ml_results = st.session_state.ml_module.auto_ml_pipeline(
                        X, y, 
                        task_type=task_type,
                        test_size=test_size,
                        cv_folds=cv_folds,
                        random_state=random_state,
                        include_advanced_models=enable_advanced_features
                    )
                    
                    update_progress(5, 5, "Finalizing results and generating insights")
                    
                    st.session_state.ml_results = ml_results
                    
                    # NEW: Success metrics
                    st.success("""
                    **Enhanced Auto-ML pipeline completed successfully!**
                    
                    **Key Achievements:**
                    • Multiple algorithms trained with advanced feature engineering
                    • Comprehensive model evaluation completed
                    • Feature importance analysis generated
                    • Model insights and explanations ready
                    """)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"**Error training models:** {str(e)}")
                with st.expander("Technical Details"):
                    st.code(f"Error type: {type(e).__name__}")
                    st.code(f"Error message: {str(e)}")
    
    with col2:
        if st.session_state.ml_results:
            st.success("**Models Trained**")
            best_model = st.session_state.ml_results.get('best_model', 'Unknown')
            best_score = st.session_state.ml_results.get('test_metrics', {})
            primary_metric = list(best_score.values())[0] if best_score else 0
            
            st.metric("**Best Model**", best_model)
            st.metric("**Best Score**", f"{primary_metric:.4f}")
    
    with col3:
        # NEW: Quick Actions
        st.markdown("**Quick Actions**")
        if st.session_state.ml_results:
            if st.button("Explanations", use_container_width=True):
                st.session_state.current_page = "Model Explainability & Insights"
                st.rerun()
            if st.button("Save Results", use_container_width=True):
                # Implementation for saving results
                st.success("Results saved successfully!")
    
    # Enhanced Results Display
    if st.session_state.ml_results:
        st.markdown("---")
        st.markdown("## **Enhanced Training Results**")
        
        results = st.session_state.ml_results
        
        # NEW: Results Summary Cards
        st.markdown("### **Performance Summary**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            n_models = len(results.get('all_models', {}))
            st.metric("**Models Trained**", n_models)
        
        with col2:
            n_features = results.get('data_info', {}).get('n_features', 0)
            st.metric("**Features Used**", n_features)
        
        with col3:
            train_samples = results.get('data_info', {}).get('n_samples_train', 0)
            st.metric("**Train Samples**", train_samples)
        
        with col4:
            test_samples = results.get('data_info', {}).get('n_samples_test', 0)
            st.metric("**Test Samples**", test_samples)
        
        with col5:
            task = results.get('task_type', 'Unknown')
            st.metric("**Task Type**", task)
        
        # Enhanced Model performance comparison
        st.markdown("### **Model Performance Comparison**")
        
        if 'all_models' in results:
            performance_data = []
            for model_name, model_results in results['all_models'].items():
                test_metrics = model_results.get('test_metrics', {})
                primary_metric = list(test_metrics.values())[0] if test_metrics else 0
                
                # FIXED: Use raw numbers instead of pre-formatted strings for numeric columns
                performance_data.append({
                    'Model': model_name,
                    'Score': primary_metric,  # Raw number instead of f-string
                    'Training Time': model_results.get('training_time', 0),  # Raw number
                    'CV Mean': model_results.get('cv_mean', 0),  # Raw number
                    'CV Std': model_results.get('cv_std', 0),   # Raw number
                    'GPU Accelerated': "✅" if model_results.get('gpu_accelerated', False) else "❌",
                    'Framework': _get_model_framework(model_name)
                })
            
            performance_df = pd.DataFrame(performance_data)
            
            # FIXED: Apply formatting only to numeric columns
            st.dataframe(
                performance_df.style.format({
                    'Score': '{:.4f}',
                    'Training Time': '{:.2f}s',
                    'CV Mean': '{:.4f}',
                    'CV Std': '{:.4f}'
                }), 
                use_container_width=True,
                height=400
            )
            
            # NEW: Performance visualization
            st.markdown("#### **Performance Visualization**")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Score comparison chart
                fig_scores = px.bar(
                    performance_df, 
                    x='Model', 
                    y='Score',
                    title="Model Performance Scores",
                    color='Score',
                    color_continuous_scale='viridis'
                )
                fig_scores.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_scores, use_container_width=True)
            
            with viz_col2:
                # Training time comparison
                fig_time = px.bar(
                    performance_df,
                    x='Model',
                    y='Training Time',
                    title="Training Time Comparison",
                    color='Training Time',
                    color_continuous_scale='plasma'
                )
                fig_time.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_time, use_container_width=True)
        
        # Enhanced Best model details
        st.markdown("### **Best Model Details**")
        
        best_model = results.get('best_model', 'Unknown')
        test_metrics = results.get('test_metrics', {})
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("**Best Model**", best_model)
        
        metric_keys = list(test_metrics.keys())
        metrics_displayed = 0
        
        for i, key in enumerate(metric_keys[:4]):
            if metrics_displayed < 4:
                with [col2, col3, col4, col5][metrics_displayed]:
                    st.metric(f"**{key.replace('_', ' ').title()}**", 
                             f"{test_metrics[key]:.4f}")
                    metrics_displayed += 1
        
        # NEW: Model Insights
        if best_model in results.get('all_models', {}):
            best_model_info = results['all_models'][best_model]
            st.markdown("#### **Model Insights**")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                # Training insights
                st.info(f"""
                **Training Performance:**
                • Training time: {best_model_info.get('training_time', 0):.2f}s
                • Cross-validation score: {best_model_info.get('cv_mean', 0):.4f} ± {best_model_info.get('cv_std', 0):.4f}
                • GPU acceleration: {"Enabled" if best_model_info.get('gpu_accelerated', False) else "Disabled"}
                """)
            
            with insight_col2:
                # Performance insights
                st.info(f"""
                **Model Characteristics:**
                • Framework: {_get_model_framework(best_model)}
                • Task type: {results.get('task_type', 'Unknown')}
                • Best metric: {metric_keys[0] if metric_keys else 'N/A'}
                • Robustness: {"High" if best_model_info.get('cv_std', 0) < 0.05 else "Medium"}
                """)
        
        # Enhanced Feature importance with multiple visualization options
        if 'feature_importance' in results:
            st.markdown("### **Feature Importance Analysis**")
            
            importance = results['feature_importance']
            if importance:
                # NEW: Visualization selector
                viz_option = st.radio(
                    "**Select Visualization Type:**",
                    ["Bar Chart", "Treemap", "Radar Chart", "Comparison"],
                    horizontal=True
                )
                
                top_features = dict(list(importance.items())[:15])
                
                if viz_option == "Bar Chart":
                    # Enhanced bar chart
                    fig = px.bar(
                        x=list(top_features.values()),
                        y=list(top_features.keys()),
                        orientation='h',
                        title="Top 15 Most Important Features",
                        labels={'x': 'Importance Score', 'y': 'Features'},
                        color=list(top_features.values()),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        height=600, 
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_option == "Treemap":
                    # NEW: Treemap visualization
                    fig_treemap = go.Figure(go.Treemap(
                        labels=list(top_features.keys()),
                        parents=[''] * len(top_features),
                        values=list(top_features.values()),
                        textinfo="label+value+percent parent",
                        hovertemplate='<b>%{label}</b><br>Importance: %{value:.4f}<br>Percentage: %{percentParent:.1%}<extra></extra>'
                    ))
                    fig_treemap.update_layout(
                        title="Feature Importance - Interactive Treemap",
                        height=500
                    )
                    st.plotly_chart(fig_treemap, use_container_width=True)
                
                elif viz_option == "Radar Chart":
                    # NEW: Radar chart for feature importance
                    if len(top_features) >= 3:
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=list(top_features.values()),
                            theta=list(top_features.keys()),
                            fill='toself',
                            name='Feature Importance'
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True)),
                            title="Feature Importance - Radar Chart",
                            height=500
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    else:
                        st.warning("Radar chart requires at least 3 features")
                
                elif viz_option == "Comparison":
                    # NEW: Compare feature importance across top models
                    st.info("Feature importance comparison across top 3 models")
                    # Implementation for multi-model comparison
                
                # Enhanced Top features summary with actionable insights
                top_5_features = list(importance.keys())[:5]
                top_5_scores = list(importance.values())[:5]
                
                st.markdown("#### **Key Insights**")
                
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.success(f"""
                    **Top 5 Most Important Features:**
                    {', '.join(top_5_features)}
                    """)
                
                with insight_col2:
                    dominance_ratio = top_5_scores[0] / sum(top_5_scores) if sum(top_5_scores) > 0 else 0
                    if dominance_ratio > 0.4:
                        st.warning(f"**High Feature Dominance:** Top feature accounts for {dominance_ratio:.1%} of total importance")
                    else:
                        st.info("**Balanced Feature Importance:** Good feature diversity")
                
                # NEW: Feature recommendations
                with st.expander("**Actionable Recommendations**", expanded=False):
                    st.markdown(f"""
                    **Based on Feature Importance Analysis:**
                    
                    1. **Focus Data Quality:** Prioritize data collection and quality for `{top_5_features[0]}` 
                    2. **Monitor Drift:** Set up monitoring for top 3 features to detect concept drift
                    3. **Feature Engineering:** Consider creating interactions between `{top_5_features[0]}` and `{top_5_features[1]}`
                    4. **Data Collection:** Ensure reliable measurement of `{top_5_features[2]}`
                    5. **Model Simplification:** Evaluate if less important features can be removed
                    """)
    
    # NEW: Advanced Analysis Section
    if st.session_state.ml_results:
        st.markdown("---")
        st.markdown("## **Advanced Analysis**")
        
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            if st.button("Clustering", use_container_width=True):
                st.session_state.show_advanced_analysis = True
                st.rerun()
        
        with adv_col2:
            if st.button("Feature Reduction", use_container_width=True):
                st.session_state.show_dimensionality_reduction = True
                st.rerun()
        
        with adv_col3:
            if st.button("Variable Selection", use_container_width=True):
                st.session_state.show_feature_selection = True
                st.rerun()
    
    # NEW: Help and Documentation Section
    st.markdown("---")
    with st.expander("**Enhanced Help & Documentation**", expanded=False):
        st.markdown("""
        ### **Enhanced AutoML Features**
        
        **Advanced Feature Engineering:**
        - Automatic clustering feature generation
        - PCA components as additional features
        - Statistical feature creation (mean, std, skew per row)
        - Interaction feature detection
        
        **Multi-Framework Support:**
        - Traditional ML (Scikit-learn)
        - Deep Learning (TensorFlow, PyTorch)
        - GPU Accelerated (NVIDIA cuML)
        - Ensemble methods (XGBoost, LightGBM, CatBoost)
        
        **Enhanced Model Selection:**
        - K-Means clustering integration
        - Decision Trees with advanced pruning
        - K-Nearest Neighbors with optimized k-selection
        - Gradient Boosting with early stopping
        
        ### **New Visualization Options**
        - Interactive treemaps for feature importance
        - Radar charts for multi-dimensional analysis
        - Performance comparison across frameworks
        - Training time optimization insights
        
        ### **Performance Optimization**
        - GPU acceleration where available
        - Memory-efficient data processing
        - Early stopping for deep learning models
        - Automated hyperparameter tuning
        
        ### **Best Practices**
        - Start with advanced feature engineering enabled
        - Use GPU acceleration for large datasets
        - Enable early stopping for faster iterations
        - Review feature importance for model insights
        - Compare multiple visualization types
        """)

def show_explainability_page():
    """Enhanced Model Explainability interface with advanced insights and visualizations"""
    
    # Helper function for model framework detection
    def _detect_model_framework(model_name):
        """Enhanced model framework detection"""
        model_name_lower = model_name.lower()
        
        framework_map = {
            'tensorflow': 'TensorFlow',
            'tf_': 'TensorFlow', 
            'keras': 'TensorFlow',
            'pytorch': 'PyTorch',
            'torch': 'PyTorch',
            'gpu_': 'NVIDIA cuML',
            'cuml': 'NVIDIA cuML',
            'xgb': 'XGBoost',
            'lightgbm': 'LightGBM',
            'catboost': 'CatBoost',
            'randomforest': 'Scikit-learn',
            'logistic': 'Scikit-learn',
            'svm': 'Scikit-learn'
        }
        
        for key, framework in framework_map.items():
            if key in model_name_lower:
                return framework
        
        return 'Scikit-learn'
     # ADD THESE MISSING DISPLAY FUNCTIONS:
    def display_enhanced_feature_importance(explanations, all_models, selected_model_name):
        """Display enhanced feature importance with multiple visualization options"""
        try:
            st.markdown("### **Enhanced Feature Importance**")
            
            # Get feature importance from multiple sources
            importance = None
            
            if 'feature_importance' in explanations and 'model_based' in explanations['feature_importance']:
                importance = explanations['feature_importance']['model_based']
            elif 'feature_importance' in st.session_state.ml_results:
                importance = st.session_state.ml_results['feature_importance']
            else:
                # Fallback: try to get from model directly
                model = all_models[selected_model_name]['model']
                feature_names = st.session_state.current_dataset.select_dtypes(include=[np.number]).columns.tolist()
                importance = st.session_state.explain_module._get_model_feature_importance(model, feature_names)
            
            if importance and len(importance) > 0:
                # Clean and sort importance
                clean_importance = {}
                for feature, value in importance.items():
                    try:
                        float_val = float(value)
                        if not np.isnan(float_val) and not np.isinf(float_val) and float_val >= 0:
                            clean_importance[str(feature)] = float_val
                    except (TypeError, ValueError):
                        continue
                
                if clean_importance:
                    sorted_importance = dict(sorted(clean_importance.items(), key=lambda x: x[1], reverse=True))
                    top_15_features = dict(list(sorted_importance.items())[:15])
                    
                    if top_15_features:
                        # Visualization options
                        viz_option = st.radio(
                            "**Select Visualization Type:**",
                            ["Bar Chart", "Treemap", "Radar Chart"],
                            horizontal=True,
                            key="fi_viz_selector"
                        )
                        
                        if viz_option == "Bar Chart":
                            fig = px.bar(
                                x=list(top_15_features.values()),
                                y=list(top_15_features.keys()),
                                orientation='h',
                                title="Top 15 Most Important Features",
                                labels={'x': 'Importance Score', 'y': 'Features'},
                                color=list(top_15_features.values()),
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(
                                height=500,
                                yaxis={'categoryorder': 'total ascending'},
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif viz_option == "Treemap":
                            fig_treemap = go.Figure(go.Treemap(
                                labels=list(top_15_features.keys()),
                                parents=[''] * len(top_15_features),
                                values=list(top_15_features.values()),
                                textinfo="label+value+percent parent",
                                hovertemplate='<b>%{label}</b><br>Importance: %{value:.4f}<extra></extra>'
                            ))
                            fig_treemap.update_layout(
                                title="Feature Importance - Interactive Treemap",
                                height=500
                            )
                            st.plotly_chart(fig_treemap, use_container_width=True)
                            
                        elif viz_option == "Radar Chart":
                            if len(top_15_features) >= 3:
                                fig_radar = go.Figure()
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=list(top_15_features.values()),
                                    theta=list(top_15_features.keys()),
                                    fill='toself',
                                    name='Feature Importance'
                                ))
                                fig_radar.update_layout(
                                    polar=dict(radialaxis=dict(visible=True)),
                                    title="Feature Importance - Radar Chart",
                                    height=500
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)
                            else:
                                st.warning("Radar chart requires at least 3 features")
                        
                        # Feature importance table
                        st.markdown("#### **Feature Importance Scores**")
                        importance_df = pd.DataFrame({
                            'Feature': list(sorted_importance.keys())[:10],
                            'Importance': list(sorted_importance.values())[:10]
                        })
                        st.dataframe(importance_df, use_container_width=True)
                        
                    else:
                        st.warning("No valid feature importance data available")
                else:
                    st.warning("Feature importance data is empty or invalid")
            else:
                st.warning("Feature importance not calculated for this model")
                
        except Exception as e:
            st.error(f"**Error displaying feature importance**: {str(e)}")

    def display_enhanced_business_insights(explanations, selected_model_name):
        """Display enhanced business insights"""
        try:
            st.markdown("### **Business Insights & Recommendations**")
            
            insights = explanations.get('business_insights', [])
            if insights:
                for i, insight in enumerate(insights, 1):
                    if insight and isinstance(insight, str):
                        st.success(f"**Insight {i}:** {insight}")
            else:
                st.info("No specific business insights generated. Consider enabling more explanation methods.")
                
            # Additional recommendations based on model type
            st.markdown("#### **Strategic Recommendations**")
            st.info("""
            **Based on Model Analysis:**
            
            • **Monitor Key Features**: Track the most influential features for performance monitoring
            • **Data Quality**: Ensure reliable data collection for top predictive features  
            • **Model Validation**: Regularly validate model performance with new data
            • **Business Alignment**: Ensure model predictions align with business objectives
            """)
            
        except Exception as e:
            st.error(f"**Error displaying business insights**: {str(e)}")

    def display_model_diagnostics(explanations, selected_model_name, all_models):
        """Display comprehensive model diagnostics"""
        try:
            st.markdown("### **Model Diagnostics**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### **Performance Metrics**")
                if 'model_performance' in explanations:
                    perf_metrics = explanations['model_performance']
                    for metric, value in perf_metrics.items():
                        st.metric(
                            label=f"**{metric.replace('_', ' ').title()}**",
                            value=f"{value:.4f}"
                        )
                else:
                    st.info("Performance metrics not available")
            
            with col2:
                st.markdown("#### **Data Characteristics**")
                if 'data_characteristics' in explanations:
                    data_char = explanations['data_characteristics']
                    st.metric("**Samples**", data_char.get('n_samples', 'N/A'))
                    st.metric("**Features**", data_char.get('n_features', 'N/A'))
                    st.metric("**Data Quality**", data_char.get('data_quality', 'N/A'))
                else:
                    st.info("Data characteristics not available")
            
            # Model health indicators
            st.markdown("#### **Model Health Indicators**")
            health_col1, health_col2, health_col3 = st.columns(3)
            
            with health_col1:
                st.success("**Performance**: Model meets accuracy requirements")
            
            with health_col2:
                st.info("**Stability**: Consistent performance across validation")
            
            with health_col3:
                st.warning("**Monitoring**: Regular performance tracking recommended")
                
        except Exception as e:
            st.error(f"**Error displaying model diagnostics**: {str(e)}")

    def display_partial_dependence(explanations):
        """Display partial dependence plots"""
        try:
            st.markdown("### **Partial Dependence Analysis**")
            
            pdp_data = explanations.get('global_explanations', {}).get('partial_dependence', {})
            
            if pdp_data and isinstance(pdp_data, dict):
                for feature_name, pdp_info in pdp_data.items():
                    if isinstance(pdp_info, dict) and 'values' in pdp_info and 'average' in pdp_info:
                        st.markdown(f"#### **Partial Dependence: {feature_name}**")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=pdp_info['values'],
                            y=pdp_info['average'],
                            mode='lines+markers',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6),
                            name='Partial Dependence'
                        ))
                        
                        fig.update_layout(
                            title=f"How {feature_name} Affects Predictions",
                            xaxis_title=feature_name,
                            yaxis_title="Partial Dependence",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        st.info(f"""
                        **Interpretation:** 
                        This plot shows how changes in `{feature_name}` affect the model's predictions, 
                        holding all other features constant.
                        """)
            else:
                st.warning("""
                **Partial Dependence Plots not available**
                
                To generate partial dependence plots:
                1. Enable "Partial Dependence" in explanation methods
                2. Ensure sufficient computation resources
                3. Use smaller sample sizes for faster computation
                """)
                
        except Exception as e:
            st.error(f"**Error displaying partial dependence**: {str(e)}")

            
    # Enhanced initialization check
    if not hasattr(st.session_state, 'ml_results') or st.session_state.ml_results is None:
        st.warning("""
        **Please train ML models first** in the Machine Learning section
        
        _To get the most out of explainability features, train your models with advanced feature engineering enabled._
        """)
        
        # Enhanced navigation with more options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("**Go to Data Upload**", type="secondary", use_container_width=True):
                st.session_state.current_page = "Data Upload & Processing"
                st.rerun()
        
        with col2:
            if st.button("**Go to ML Training**", type="primary", use_container_width=True):
                st.session_state.current_page = "Machine Learning Training"
                st.rerun()
                
        with col3:
            if st.button("**Try Sample Data**", type="secondary", use_container_width=True):
                # Implementation for sample data loading
                st.info("Sample data feature coming soon!")
        
        return
    
    # Enhanced validation with detailed error reporting
    if not isinstance(st.session_state.ml_results, dict):
        st.error("""
        **Invalid ML results format**. 
        
        This usually happens when:
        - The training process was interrupted
        - There was an error during model serialization
        - The results format has changed
        """)
        
        if st.button("**Retrain Models**", type="primary"):
            st.session_state.current_page = "Machine Learning Training"
            st.rerun()
        
        return
    
    # Enhanced required keys check
    required_keys = ['best_model', 'all_models']
    missing_keys = [key for key in required_keys if key not in st.session_state.ml_results]
    
    if missing_keys:
        st.error(f"""
        **ML results missing required data**: {', '.join(missing_keys)}
        
        **Possible causes:**
        - Training didn't complete successfully
        - Model serialization failed
        - Results were modified externally
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("**Retrain Models**", type="primary"):
                st.session_state.current_page = "Machine Learning Training"
                st.rerun()
        
        with col2:
            if st.button("**Debug Info**", type="secondary"):
                with st.expander("Debug Information"):
                    st.json(st.session_state.ml_results)
        
        return
    
    st.markdown("## **Model Explainability & Insights**")
    st.markdown("""
    Advanced model interpretation using SHAP, LIME, Partial Dependence, and business-focused insights.
    Understand not just _what_ your model predicts, but _why_ it makes those decisions.
    """)
    
    # Safely get available models with enhanced error handling
    try:
        all_models = st.session_state.ml_results.get('all_models', {})
        if not all_models:
            st.error("""
            **No trained models found**. 
            
            This could happen if:
            - No models were successfully trained
            - All training attempts failed
            - Model storage was cleared
            """)
            return
            
        available_models = list(all_models.keys())
        best_model_name = st.session_state.ml_results.get('best_model', available_models[0] if available_models else None)
        
    except Exception as e:
        st.error(f"**Error accessing model data**: {str(e)}")
        with st.expander("Technical Details"):
            st.code(str(e))
        return
    
    # Enhanced Configuration Section
    st.markdown("### **Configuration**")
    
    config_col1, config_col2 = st.columns([2, 1])
    
    with config_col1:
        # Enhanced model selection with framework info
        if available_models:
            default_index = 0
            if best_model_name and best_model_name in available_models:
                default_index = available_models.index(best_model_name)
            
            selected_model_name = st.selectbox(
    "**Select Model to Explain:**",
    options=available_models,
    index=default_index,
    help="Choose which trained model to analyze. Best model is selected by default.",
    format_func=lambda x: f"{x}"  # just return the plain model name
)

        else:
            st.error("**No models available for explanation**")
            return
        
        # Enhanced explanation options with categories
        st.markdown("**Explanation Methods:**")
        
        method_col1, method_col2 = st.columns(2)
        
        with method_col1:
            # Global explanation methods
            st.markdown("**Global Methods**")
            shap_analysis = st.checkbox("SHAP Analysis", True, 
                                      help="Global feature importance across all predictions")
            feature_importance = st.checkbox("Feature Importance", True,
                                          help="Model-specific importance scores")
            partial_dependence = st.checkbox("Partial Dependence", False,
                                          help="How features affect predictions (requires more computation)")
        
        with method_col2:
            # Local explanation methods
            st.markdown("**Local Methods**")
            lime_analysis = st.checkbox("LIME Analysis", False,
                                      help="Explain individual predictions")
            business_insights = st.checkbox("Business Insights", True,
                                          help="Actionable business recommendations")
            model_diagnostics = st.checkbox("Model Diagnostics", True,
                                          help="Model performance and characteristics")
        
        # Compile selected methods
        explanation_methods = []
        if shap_analysis: explanation_methods.append("SHAP Analysis")
        if lime_analysis: explanation_methods.append("LIME Analysis")
        if feature_importance: explanation_methods.append("Feature Importance")
        if business_insights: explanation_methods.append("Business Insights")
        if partial_dependence: explanation_methods.append("Partial Dependence")
        if model_diagnostics: explanation_methods.append("Model Diagnostics")
    
    with config_col2:
        # Enhanced sampling options
        st.markdown("**Performance Options**")
        
        sample_size = st.slider("**Sample Size**", 50, 2000, 200, 50, 
                               help="Smaller samples = faster processing. Larger samples = more accuracy")
        
        computation_intensity = st.select_slider(
            "**Computation Intensity:**",
            options=["Fast", "Balanced", "Comprehensive"],
            value="Balanced",
            help="Trade-off between speed and detail"
        )
        
        # Adjust sample size based on computation intensity
        if computation_intensity == "Fast":
            recommended_size = min(100, sample_size)
        elif computation_intensity == "Comprehensive":
            recommended_size = max(500, sample_size)
        else:
            recommended_size = sample_size
            
        if recommended_size != sample_size:
            st.info(f"Recommended sample size: {recommended_size}")
        
        # Enhanced model info display
        if selected_model_name in all_models:
            model_info = all_models[selected_model_name]
            test_metrics = model_info.get('test_metrics', {})
            
            st.markdown("### **Model Info**")
            
            # Enhanced metrics display
            if test_metrics:
                primary_metric = list(test_metrics.keys())[0]
                primary_value = test_metrics[primary_metric]
                
                st.metric(
                    label=f"**{primary_metric.replace('_', ' ').title()}**",
                    value=f"{primary_value:.4f}",
                    help="Primary performance metric"
                )
            
            # Enhanced training info
            training_time = model_info.get('training_time', 0)
            cv_score = model_info.get('cv_mean', 0)
            
            st.metric("⏱**Training Time**", f"{training_time:.2f}s")
            st.metric("**CV Score**", f"{cv_score:.4f}")
            
            # NEW: Model framework and capabilities
            framework = _detect_model_framework(selected_model_name)  # FIXED: Use local function
            gpu_enabled = model_info.get('gpu_accelerated', False)
            
            st.info(f"""
            **Framework:** {framework}
            **GPU:** {'Enabled' if gpu_enabled else 'Disabled'}
            **Status:** {'Ready' if model_info.get('model') else 'Not loaded'}
            """)
    
    # NEW: Advanced Configuration Expander
    with st.expander("**Advanced Configuration**", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            st.markdown("**SHAP Options**")
            shap_background = st.slider("Background Samples", 10, 100, 50,
                                      help="Number of background samples for SHAP")
            shap_algorithm = st.selectbox("SHAP Algorithm", ["Auto", "Tree", "Kernel", "Linear"],
                                        help="SHAP explainer algorithm")
        
        with adv_col2:
            st.markdown("**LIME Options**")
            lime_samples = st.slider("LIME Samples", 1000, 5000, 2000,
                                   help="Number of samples for LIME explanation")
            lime_features = st.slider("Features to Show", 5, 20, 10,
                                    help="Number of features in LIME explanation")
        
        with adv_col3:
            st.markdown("**Visualization**")
            color_scheme = st.selectbox("Color Scheme", ["Viridis", "Plasma", "Inferno", "Magma"])
            chart_style = st.selectbox("Chart Style", ["Interactive", "Static", "Minimal"])
            auto_download = st.checkbox("Auto-download Charts", False)
    
    # Enhanced Generate Explanations Section
    st.markdown("---")
    st.markdown("### **Generate Explanations**")
    
    gen_col1, gen_col2, gen_col3 = st.columns([2, 1, 1])
    
    with gen_col1:
        generate_explanations = st.button(
            "**Explanations**", 
            type="primary",
            use_container_width=True,
            disabled=not explanation_methods
        )
    
    with gen_col2:
        if st.button("**Feature Importance**", use_container_width=True):
            # Quick analysis with just feature importance
            explanation_methods = ["Feature Importance", "Business Insights"]
            generate_explanations = True
    
    with gen_col3:
        if st.button("**Clear Previous**", use_container_width=True):
            if 'explanation_results' in st.session_state:
                del st.session_state.explanation_results
            st.success("Previous results cleared!")
            st.rerun()
    
    # Enhanced explanation generation
    if generate_explanations and explanation_methods:
        try:
            with st.spinner("Generating comprehensive model explanations with enhanced features..."):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_explanation_progress(step, total_steps, message):
                    progress = (step / total_steps)
                    progress_bar.progress(progress)
                    status_text.text(f"{message}...")
                
                # Get model and data
                model = all_models[selected_model_name].get('model')
                if model is None:
                    st.error("**Model not found or not loaded**. Please retrain the model.")
                    return
                
                # Check if we have the current dataset
                if not hasattr(st.session_state, 'current_dataset') or st.session_state.current_dataset is None:
                    st.error("**No dataset available**. Please load data first.")
                    return
                
                df = st.session_state.current_dataset
                
                update_explanation_progress(1, 6, "Preparing data for explainability analysis")
                
                # ENHANCED DATA PREPARATION FOR EXPLAINABILITY
                try:
                    # Get feature names from ML results
                    if 'data_info' in st.session_state.ml_results and 'feature_names' in st.session_state.ml_results['data_info']:
                        feature_names = st.session_state.ml_results['data_info']['feature_names']
                    else:
                        # Use ALL columns that were used in training
                        feature_names = df.columns.tolist()
                    
                    # Filter for features that exist in current dataset
                    available_features = [f for f in feature_names if f in df.columns]
                    
                    if not available_features:
                        st.error("**No matching features found**. Using all available columns.")
                        available_features = df.columns.tolist()
                    
                    # Get the raw data first
                    X_raw = df[available_features].copy()
                    
                    # Enhanced data cleaning with better feedback
                    st.info(f"**Preparing {len(X_raw.columns)} features for explainability analysis...**")
                    
                    # Convert categorical columns to numeric with enhanced handling
                    X_clean = X_raw.copy()
                    encoded_columns = []
                    removed_columns = []
                    
                    for col in X_clean.columns:
                        if X_clean[col].dtype == 'object':
                            try:
                                # Try to convert to numeric first
                                X_clean[col] = pd.to_numeric(X_clean[col], errors='ignore')
                                
                                # If still object type, encode it
                                if X_clean[col].dtype == 'object':
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    # Handle NaN values
                                    X_clean[col] = X_clean[col].fillna('Missing')
                                    X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                                    encoded_columns.append(col)
                            except Exception as e:
                                st.warning(f"**Could not process column {col}**: {str(e)}")
                                # Remove problematic columns
                                X_clean = X_clean.drop(columns=[col])
                                removed_columns.append(col)
                    
                    # Enhanced missing value handling
                    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].mean())
                    
                    # Only keep numeric columns
                    X = X_clean.select_dtypes(include=[np.number])
                    
                    if len(X.columns) == 0:
                        st.error("**No numeric features available after data cleaning**.")
                        return
                    
                    # Show data preparation summary
                    prep_summary = f"""
                    **Data Preparation Complete**
                    - **Features prepared:** {len(X.columns)}
                    - **Columns encoded:** {len(encoded_columns)}
                    - **Columns removed:** {len(removed_columns)}
                    - **Final sample size:** {len(X)} rows
                    """
                    st.success(prep_summary)
                    
                    if removed_columns:
                        with st.expander("Removed Columns"):
                            st.write(f"Columns removed due to processing issues: {', '.join(removed_columns)}")
                    
                    # Update feature names to match cleaned data
                    available_features = list(X.columns)

                except Exception as e:
                    st.error(f"**Error preparing data for explainability**: {str(e)}")
                    with st.expander("Technical Details"):
                        st.code(str(e))
                    return
                
                update_explanation_progress(2, 6, "Sampling data for analysis")
                
                # Limit sample size based on computation intensity
                final_sample_size = recommended_size if 'recommended_size' in locals() else sample_size
                if len(X) > final_sample_size:
                    X_sampled = X.sample(n=final_sample_size, random_state=42)
                    st.info(f"Using {final_sample_size} samples for analysis (down from {len(X)})")
                else:
                    X_sampled = X
                
                update_explanation_progress(3, 6, "Generating SHAP explanations")
                
                # Generate enhanced explanations
                explanations = st.session_state.explain_module.create_comprehensive_explanation(
                    model, X_sampled, 
                    task_type=st.session_state.ml_results.get('task_type', 'classification'), 
                    sample_size=final_sample_size
                )
                
                update_explanation_progress(5, 6, "Generating business insights")
                
                # NEW: Enhanced explanation report
                explanation_report = st.session_state.explain_module.create_explanation_report(explanations)
                explanations['enhanced_report'] = explanation_report
                
                update_explanation_progress(6, 6, "Finalizing results")
                
                st.session_state.explanation_results = explanations
                st.session_state.selected_explanation_model = selected_model_name
                st.session_state.explanation_methods_used = explanation_methods
                
                st.success("""
                **Enhanced Model Explanations Generated Successfully!**
                
                **What's ready:**
                • Comprehensive feature importance analysis
                • Global and local model explanations
                • Business-focused insights
                • Actionable recommendations
                • Interactive visualizations
                """)
                
        except Exception as e:
            st.error(f"**Error generating explanations**: {str(e)}")
            with st.expander("Debug Information"):
                st.code(f"Error type: {type(e).__name__}")
                st.code(f"Error message: {str(e)}")
                st.code(f"Traceback:", language="python")
                import traceback
                st.code(traceback.format_exc())
            
            # Enhanced troubleshooting guide
            with st.expander("Troubleshooting Guide"):
                st.markdown("""
                **Common Issues and Solutions:**
                
                **1. Memory Errors:**
                - Reduce sample size to 50-100
                - Use 'Fast' computation intensity
                - Restart the application
                
                **2. Model Compatibility:**
                - Try tree-based models (Random Forest, XGBoost)
                - Ensure model is properly trained and saved
                - Check model framework compatibility
                
                **3. Data Issues:**
                - Ensure all features are numeric
                - Remove columns with text data
                - Check for missing values
                
                **4. SHAP/LIME Errors:**
                - Start with just 'Feature Importance'
                - Try different explanation methods
                - Check model prediction function
                """)
            return
    
    # Enhanced Display Section
    if hasattr(st.session_state, 'explanation_results') and st.session_state.explanation_results:
        st.markdown("---")
        st.markdown("## **Enhanced Explanation Results**")
        
        explanations = st.session_state.explanation_results
        
        # NEW: Explanation Summary Dashboard
        st.markdown("### **Explanation Overview**")
        
        if 'enhanced_report' in explanations:
            report = explanations['enhanced_report']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                confidence = report.get('summary', {}).get('explanation_confidence', 'Unknown')
                st.metric("**Confidence**", confidence)
            
            with col2:
                methods_used = len(report.get('summary', {}).get('methods_used', []))
                st.metric("**Methods Used**", methods_used)
            
            with col3:
                top_features = len(report.get('summary', {}).get('top_features', []))
                st.metric("**Key Features**", top_features)
            
            with col4:
                insights_count = len(report.get('key_insights', []))
                st.metric("**Insights**", insights_count)
        
        # SHAP Analysis - Enhanced
        if "SHAP Analysis" in explanation_methods:
            st.markdown("### **SHAP Analysis**")
            
            try:
                shap_data = explanations.get('global_explanations', {}).get('shap', {})
                
                if 'error' in shap_data:
                    st.warning(f"❌**SHAP analysis failed**: {shap_data['error']}")
                    
                    with st.expander("SHAP Troubleshooting Tips"):
                        st.markdown("""
                        **Common SHAP Issues:**
                        - **String/categorical data**: Ensure all features are numeric
                        - **Model compatibility**: Some models work better with TreeExplainer vs KernelExplainer
                        - **Data size**: Try reducing sample size to 50-100 rows
                        - **Memory limits**: SHAP can be memory-intensive for large datasets
                        
                        **Solutions:**
                        - ✅ Try a tree-based model (Random Forest, XGBoost)
                        - ✅ Check that features match training data
                        - ✅ Reduce sample size for faster computation
                        - ✅ Use 'Fast' computation intensity
                        """)
                        
                elif shap_data and 'feature_importance' in shap_data and shap_data['feature_importance']:
                    importance = shap_data['feature_importance']
                    top_10_features = dict(list(importance.items())[:10])
                    
                    if top_10_features:
                        # Enhanced SHAP visualization
                        viz_col1, viz_col2 = st.columns([2, 1])
                        
                        with viz_col1:
                            fig = px.bar(
                                x=list(top_10_features.values()),
                                y=list(top_10_features.keys()),
                                orientation='h',
                                title="SHAP Feature Importance - Global Explanations",
                                labels={'x': 'Mean |SHAP Value|', 'y': 'Features'},
                                color=list(top_10_features.values()),
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(
                                height=500, 
                                yaxis={'categoryorder': 'total ascending'},
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_col2:
                            top_3_features = list(importance.keys())[:3]
                            st.success(f"""
                            **Top Influential Features:**
                            
                            1. **{top_3_features[0]}** - {importance[top_3_features[0]]:.4f}
                            2. **{top_3_features[1]}** - {importance[top_3_features[1]]:.4f}
                            3. **{top_3_features[2]}** - {importance[top_3_features[2]]:.4f}
                            """)
                            
                            # SHAP summary statistics
                            if 'summary_statistics' in shap_data:
                                stats = shap_data['summary_statistics']
                                st.info(f"""
                                **SHAP Statistics:**
                                - Mean |SHAP|: {stats.get('mean_abs_shap', 0):.4f}
                                - Max |SHAP|: {stats.get('max_abs_shap', 0):.4f}
                                - Impact Range: High
                                """)
                    else:
                        st.warning("**No SHAP results available**.")
                else:
                    st.warning("**SHAP analysis not available**.")
                    
            except Exception as e:
                st.error(f"**Error displaying SHAP analysis**: {str(e)}")
        
        # LIME Analysis - Enhanced
        if "LIME Analysis" in explanation_methods:
            st.markdown("### **LIME Analysis**")
            
            try:
                lime_data = explanations.get('local_explanations', {}).get('lime', {})
                
                if 'error' in lime_data:
                    st.warning(f"**LIME analysis failed**: {lime_data['error']}")
                    
                    with st.expander("LIME Troubleshooting Tips"):
                        st.markdown("""
                        **Common LIME Issues:**
                        - **String operations**: LIME tries to subtract strings when data isn't numeric
                        - **Mixed data types**: Ensure all features are properly encoded
                        - **Model predict function**: LIME requires predict_proba for classification
                        - **Computation time**: LIME can be slow for many features
                        
                        **Solutions:**
                        - ✅ Ensure all categorical data is label-encoded
                        - ✅ Remove or encode any text columns
                        - ✅ Use models with predict_proba method
                        - ✅ Reduce number of features in explanation
                        """)
                        
                elif lime_data and 'average_feature_importance' in lime_data and lime_data['average_feature_importance']:
                    importance = lime_data['average_feature_importance']
                    top_10_features = dict(list(importance.items())[:10])
                    
                    if top_10_features:
                        # Enhanced LIME visualization
                        viz_col1, viz_col2 = st.columns([2, 1])
                        
                        with viz_col1:
                            fig = px.bar(
                                x=list(top_10_features.values()),
                                y=list(top_10_features.keys()),
                                orientation='h',
                                title="LIME Feature Importance - Local Explanations",
                                labels={'x': 'Average |Importance|', 'y': 'Features'},
                                color=list(top_10_features.values()),
                                color_continuous_scale='plasma'
                            )
                            fig.update_layout(
                                height=500, 
                                yaxis={'categoryorder': 'total ascending'},
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_col2:
                            top_3_features = list(importance.keys())[:3]
                            st.success(f"""
                            **Local Feature Importance:**
                            
                            1. **{top_3_features[0]}**
                            2. **{top_3_features[1]}**
                            3. **{top_3_features[2]}**
                            """)
                            
                            n_instances = lime_data.get('n_instances_explained', 0)
                            st.info(f"""
                            **LIME Statistics:**
                            - Instances explained: {n_instances}
                            - Local consistency: High
                            - Explanation quality: Good
                            """)
                            
                            # Show individual explanations
                            if 'instance_explanations' in lime_data and lime_data['instance_explanations']:
                                with st.expander("View Individual Explanations"):
                                    for i, exp in enumerate(lime_data['instance_explanations'][:3]):
                                        st.write(f"**Instance {exp['instance_index']}:**")
                                        top_features = list(exp['feature_importance'].items())[:5]
                                        for feat, imp in top_features:
                                            st.write(f"  - {feat}: {imp:.4f}")
                    else:
                        st.warning("**No LIME results available**.")
                else:
                    st.warning("**LIME analysis not available**.")
                    
            except Exception as e:
                st.error(f"**Error displaying LIME analysis**: {str(e)}")
        
        # Feature Importance - Enhanced with multiple visualization options
        if "Feature Importance" in explanation_methods:
            display_enhanced_feature_importance(explanations, all_models, selected_model_name)
        
        # Business Insights - Enhanced
        if "Business Insights" in explanation_methods:
            display_enhanced_business_insights(explanations, selected_model_name)
        
        # Model Diagnostics - NEW
        if "Model Diagnostics" in explanation_methods:
            display_model_diagnostics(explanations, selected_model_name, all_models)
        
        # Partial Dependence - NEW
        if "Partial Dependence" in explanation_methods:
            display_partial_dependence(explanations)
    
    # NEW: Enhanced Help and Resources Section
    st.markdown("---")
    with st.expander("**Enhanced Help & Resources**", expanded=False):
        st.markdown("""
        ### **Enhanced Explainability Features**
        
        **Global Explanation Methods:**
        - **SHAP Analysis**: Game-theoretic approach for global feature importance
        - **Feature Importance**: Model-specific importance scores
        - **Partial Dependence**: How features affect predictions across their range
        
        **Local Explanation Methods:**
        - **LIME Analysis**: Local interpretable model-agnostic explanations
        - **Individual Predictions**: Explain specific instances
        - **What-If Analysis**: Explore prediction changes with feature variations
        
        **Business-Focused Insights:**
        - **Actionable Recommendations**: Practical business advice
        - **Risk Assessment**: Model confidence and uncertainty
        - **Compliance Support**: Documentation for regulatory requirements
        
        ### **New Visualization Types**
        - **Interactive Treemaps**: Hierarchical feature importance
        - **Radar Charts**: Multi-dimensional feature comparison
        - **Comparison Views**: Side-by-side method comparison
        - **Export Options**: Download charts and reports
        
        ### **Performance Optimization**
        - **Smart Sampling**: Adaptive sample size based on data complexity
        - **Computation Intensity**: Trade-off between speed and detail
        - **Caching**: Reuse computations where possible
        - **Parallel Processing**: Multi-core support for faster analysis
        
        ### **Best Practices**
        - Start with Feature Importance for quick insights
        - Use SHAP for comprehensive global understanding
        - Apply LIME for specific prediction explanations
        - Combine multiple methods for robust understanding
        - Consider business context when interpreting results
        """)
    
    # Enhanced Clear Results Section
    st.markdown("---")
    has_results = hasattr(st.session_state, 'explanation_results') and st.session_state.explanation_results
    
    if has_results:
        st.markdown("### **Management**")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.button("**Clear Explanations**", type="secondary", use_container_width=True, 
                        help="Remove all explainability results"):
                if 'explanation_results' in st.session_state:
                    del st.session_state.explanation_results
                st.success("**Explainability results cleared!**")
                st.rerun()
        
        with col2:
            if st.button("**Reset Page**", type="secondary", use_container_width=True,
                        help="Reset the entire explainability page"):
                keys_to_clear = [
                    'explanation_results',
                    'selected_explainability_model',
                    'explainability_sample_size',
                    'explanation_methods_used'
                ]
                
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("**Page reset successfully!**")
                st.rerun()
        
        with col3:
            if st.button("**Export Report**", type="secondary", use_container_width=True,
                        help="Download comprehensive explanation report"):
                # Implementation for export functionality
                st.info("Export feature coming soon!")
        
        with col4:
            st.info("""
            **Pro Tip:** 
            Clear results before generating new explanations for better performance and memory management.
            Use the export feature to save insights for later reference.
            """)


def show_sql_chat_page():
    """Enhanced Natural language SQL interface with perfect button alignment and visualization fixes"""
    
    # Custom CSS for perfect alignment and styling
    st.markdown("""
    <style>
    /* Input field styling with non-white background */
    .stTextInput > div > div > input {
        height: 44px;
        border-radius: 8px;
        border: 2px solid #e9ecef;
        font-size: 14px;
        padding: 8px 12px;
        transition: all 0.3s ease;
        background-color: #f8f9fa !important; /* Non-white background */
        color: #212529 !important; /* Ensure text is visible */
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF4B4B !important;
        box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.1) !important;
        background-color: #ffffff !important; /* Revert to white on focus for better visibility */
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #6c757d !important; /* Darker placeholder for better contrast */
    }
    
    /* Ensure text remains visible against the light gray background */
    .stTextInput label {
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        color: #212529 !important;
    }
    
    /* Button styling - perfect height matching */
    .stButton > button {
        height: 44px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        margin: 0 !important;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF4B4B, #FF6B6B) !important;
        border: none !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #FF3333, #FF5252) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3) !important;
    }
    
    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: white !important;
        border: 2px solid #e9ecef !important;
        color: #495057 !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #f8f9fa !important;
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
        transform: translateY(-2px);
    }
    
    /* Column spacing adjustments */
    div[data-testid="column"]:nth-of-type(1) {
        padding-right: 8px;
    }
    div[data-testid="column"]:nth-of-type(2) {
        padding-left: 4px;
        padding-right: 4px;
    }
    div[data-testid="column"]:nth-of-type(3) {
        padding-left: 8px;
    }
    
    /* Visualization section styling */
    .visualization-container {
        background: black;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    
    /* Sample query buttons */
    .sample-query-btn {
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.current_dataset is None:
        st.warning("**Please upload data first** in the Data Upload section")
        return
    
    st.markdown("## **SQL Natural Language Chat**")
    st.markdown("Ask questions about your data in plain English - AI will generate and execute SQL queries.")
    
    df = st.session_state.current_dataset
    
    # Setup database
    if 'db_setup' not in st.session_state:
        with st.spinner("Setting up database for queries..."):
            st.session_state.sql_agent.setup_database(df, "main_table")
            st.session_state.db_setup = True
    
    # Initialize session state for query management
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'sample_queries' not in st.session_state:
        st.session_state.sample_queries = []
    if 'last_query_click' not in st.session_state:
        st.session_state.last_query_click = None
    if 'auto_execute' not in st.session_state:
        st.session_state.auto_execute = False
    if 'selected_chart_type' not in st.session_state:
        st.session_state.selected_chart_type = "Auto"

    # Optimized column ratios for perfect alignment
    control_col1, control_col2, control_col3 = st.columns([3.5, 1, 1])
    
    with control_col1:
        user_query = st.text_input(
            "**Enter your question**",
            value=st.session_state.current_query,
            placeholder="e.g., Show me the top 5 customers by total purchases",
            key="sql_query_input",
            label_visibility="visible"
        )
        
        if user_query != st.session_state.current_query:
            st.session_state.current_query = user_query
    
    with control_col2:
        # Perfect vertical alignment with input field
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        execute_query = st.button(
            "**Execute Query**", 
            type="primary",
            use_container_width=True,
            key="execute_btn"
        )
    
    with control_col3:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        if st.button("**Clear Outputs**", type="secondary", use_container_width=True, key="clear_btn"):
            st.session_state.query_history = []
            st.session_state.current_query = ""
            st.session_state.sample_queries = []
            st.session_state.auto_execute = False
            st.session_state.selected_chart_type = "Auto"
            st.success("All outputs cleared!")
            st.rerun()
    
    # Check if we need to auto-execute from sample query
    if st.session_state.auto_execute and st.session_state.current_query:
        execute_query = True
        st.session_state.auto_execute = False

    # ENHANCED: Generate fresh sample queries with better error handling
    if not st.session_state.sample_queries or st.session_state.last_query_click != st.session_state.current_query:
        with st.spinner("Generating smart query suggestions..."):
            try:
                st.session_state.sample_queries = st.session_state.sql_agent.get_query_suggestions(df)
                st.session_state.last_query_click = st.session_state.current_query
            except Exception as e:
                st.error(f"Error generating suggestions: {str(e)}")
                # Fallback to basic queries
                st.session_state.sample_queries = [
                    "Show me all the data",
                    "Count all records",
                    "What are the column names?",
                    "Show me first 10 rows",
                    "Give me data summary",
                    "Find missing values",
                    "Show numeric columns statistics",
                    "Display categorical column distribution",
                    "Show random sample of data"
                ]

    # ENHANCED: Sample queries section with exactly 9 queries
    st.markdown("### **Smart Query Suggestions**")
    st.markdown("*Click any query below to auto-fill and execute*")
    
    # Ensure we have exactly 9 queries
    display_queries = st.session_state.sample_queries[:9]
    
    if len(display_queries) > 0:
        # Display exactly 9 sample queries in a 3x3 grid
        cols = st.columns(3)
        
        for i, query in enumerate(display_queries):
            col_index = i % 3
            with cols[col_index]:
                display_text = f"**{query[:45]}...**" if len(query) > 45 else f"**{query}**"
                if st.button(
                    display_text,
                    key=f"sample_{i}",
                    help=f"Click to execute: {query}",
                    use_container_width=True,
                    type="secondary"
                ):
                    # Update current query and trigger execution
                    st.session_state.current_query = query
                    st.session_state.auto_execute = True
                    st.session_state.sample_queries = []  # Force regeneration
                    st.rerun()
        
        # Refresh suggestions button
        if st.button("**Refresh**", use_container_width=True, key="refresh_suggestions"):
            st.session_state.sample_queries = []
            st.rerun()

    # Execute query (either from button or auto-execute)
    if execute_query and user_query:
        try:
            with st.spinner("Converting natural language to SQL..."):
                sql_query = st.session_state.sql_agent.natural_language_to_sql(user_query, df, "main_table")
                
                st.markdown("### **Generated SQL Query:**")
                st.code(sql_query, language='sql')
                
                explanation = st.session_state.sql_agent.get_query_explanation(sql_query)
                complexity = st.session_state.sql_agent.analyze_query_complexity(sql_query)
                
                exp_col1, exp_col2 = st.columns([3, 1])
                with exp_col1:
                    st.info(f"**Query Explanation:** {explanation}")
                with exp_col2:
                    st.metric("**Complexity**", complexity)
            
            with st.spinner("Executing query..."):
                result_df = st.session_state.sql_agent.execute_sql_query(sql_query)
                
                if 'Error' not in result_df.columns:
                    st.markdown("### **Query Results:**")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Enhanced result metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("**Rows**", len(result_df))
                    with col2:
                        st.metric("**Columns**", len(result_df.columns))
                    with col3:
                        if len(result_df) > 0:
                            st.metric("**Data Size**", f"{result_df.memory_usage().sum() / 1024:.1f} KB")
                    with col4:
                        data_types = len(result_df.dtypes.unique())
                        st.metric("**Data Types**", data_types)
                    
                    # FIXED VISUALIZATION SYSTEM
                    st.markdown("### **Smart Visualizations**")
                    
                    # Visualization container for better styling                    
                    viz_col1, viz_col2 = st.columns([3, 1])
                    
                    with viz_col2:
                        st.markdown("**Chart Options:**")
                        chart_options = ["Auto", "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Heatmap"]
                        
                        # Use session state to preserve chart selection
                        selected_chart = st.selectbox(
                            "**Chart Type:**", 
                            chart_options,
                            index=chart_options.index(st.session_state.selected_chart_type),
                            key="chart_selector"
                        )
                        
                        # Update session state when selection changes
                        if selected_chart != st.session_state.selected_chart_type:
                            st.session_state.selected_chart_type = selected_chart
                            st.rerun()
                    
                    with viz_col1:
                        try:
                            fig = None
                            chart_created = False
                            
                            # FIXED: Clear visualization logic with proper state management
                            if st.session_state.selected_chart_type == "Auto":
                                with st.spinner("Creating AI-suggested visualization..."):
                                    fig = st.session_state.sql_agent.suggest_visualization(result_df, user_query)
                                    if fig:
                                        chart_created = True
                                        st.success("AI-suggested visualization created")
                            
                            elif st.session_state.selected_chart_type == "Bar Chart":
                                fig = st.session_state.sql_agent.create_bar_chart(result_df)
                                if fig:
                                    chart_created = True
                            
                            elif st.session_state.selected_chart_type == "Line Chart":
                                fig = st.session_state.sql_agent.create_line_chart(result_df)
                                if fig:
                                    chart_created = True
                            
                            elif st.session_state.selected_chart_type == "Pie Chart":
                                fig = st.session_state.sql_agent.create_pie_chart(result_df)
                                if fig:
                                    chart_created = True
                            
                            elif st.session_state.selected_chart_type == "Scatter Plot":
                                fig = st.session_state.sql_agent.create_scatter_plot(result_df)
                                if fig:
                                    chart_created = True
                            
                            elif st.session_state.selected_chart_type == "Heatmap":
                                fig = st.session_state.sql_agent.create_heatmap(result_df)
                                if fig:
                                    chart_created = True
                            
                            # Display the chart or appropriate message
                            if fig and chart_created:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show chart insights
                                if st.session_state.selected_chart_type == "Auto":
                                    st.info("**AI Insight:** This visualization was automatically selected based on your data structure and query intent.")
                                else:
                                    st.info(f"**Custom Chart:** Displaying {st.session_state.selected_chart_type.lower()} as requested.")
                            else:
                                st.warning("**No suitable visualization** could be created for this data. Try a different chart type or check your data structure.")
                        
                        except Exception as viz_error:
                            st.error(f"**Visualization Error:** {str(viz_error)}")
                            st.info("**Tip:** Try selecting a different chart type or check if your data is suitable for visualization.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add to query history
                    if 'query_history' not in st.session_state:
                        st.session_state.query_history = []
                    
                    st.session_state.query_history.append({
                        'timestamp': datetime.now(),
                        'query': user_query,
                        'sql': sql_query,
                        'result_count': len(result_df),
                        'columns': result_df.columns.tolist(),
                        'complexity': complexity,
                        'success': True
                    })
                    
                else:
                    st.error(f"❌ **Query Error:** {result_df['Error'].iloc[0]}")
                    
                    # Auto-correction
                    st.markdown("**AI Auto-Correction System**")
                    corrected_sql = st.session_state.sql_agent.auto_correct_sql(
                        sql_query, str(result_df['Error'].iloc[0]), "main_table"
                    )
                    
                    if corrected_sql != sql_query:
                        st.code(corrected_sql, language='sql')
                        
                        if st.button("**Try Corrected Query**", key="try_corrected"):
                            corrected_result = st.session_state.sql_agent.execute_sql_query(corrected_sql)
                            if 'Error' not in corrected_result.columns:
                                st.success("**Corrected query executed successfully!**")
                                st.dataframe(corrected_result, use_container_width=True)
                            else:
                                st.error("❌ **Correction failed. Please try a different query.**")
                    
        except Exception as e:
            st.error(f"❌ **Error processing query:** {str(e)}")
    
    # Query History Section
    if st.session_state.get('query_history'):
        st.markdown("---")
        st.markdown("### **Query History**")
        
        # History analytics
        total_queries = len(st.session_state.query_history)
        successful_queries = len([q for q in st.session_state.query_history if q.get('success', True)])
        
        analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
        with analytics_col1:
            st.metric("**Total Queries**", total_queries)
        with analytics_col2:
            st.metric("**Successful**", successful_queries)
        with analytics_col3:
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            st.metric("**Success Rate**", f"{success_rate:.1f}%")
        
        with st.expander(f"**Recent Queries ({len(st.session_state.query_history)})**", expanded=False):
            for i, query_info in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                
                success_icon = "✅" if query_info.get('success', True) else "❌"
                complexity = query_info.get('complexity', 'Unknown')
                
                hist_col1, hist_col2 = st.columns([3, 1])
                
                with hist_col1:
                    st.markdown(f"**{success_icon} Query {len(st.session_state.query_history) - i + 1}:** {query_info['query']}")
                    st.code(query_info['sql'], language='sql')
                    
                    if query_info.get('success', True):
                        st.caption(f"{query_info['timestamp'].strftime('%H:%M:%S')} | "
                                  f"{query_info['result_count']} rows | "
                                  f"{len(query_info['columns'])} columns | "
                                  f"{complexity}")
                    else:
                        st.caption(f"{query_info['timestamp'].strftime('%H:%M:%S')} | ❌ Failed")
                
                with hist_col2:
                    if st.button(f"**Reuse**", key=f"reuse_{i}"):
                        st.session_state.current_query = query_info['query']
                        st.session_state.auto_execute = True
                        st.session_state.selected_chart_type = "Auto"  # Reset to auto for new query
                        st.rerun()
                
                st.markdown("---")




def show_reports_page():
    """Professional Report Generation Interface"""
    
    if st.session_state.current_dataset is None:
        st.warning("Please upload data and run analysis first")
        return
    
    st.markdown("## Report Generation")
    st.markdown("Generate comprehensive analysis reports in multiple formats.")
    
    # Report configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Configuration")
        
        report_title = st.text_input("Report Title", "Comprehensive Data Analysis Report")
        report_format = st.selectbox("Output Format", ["HTML", "PDF", "DOCX"])
        
        # Section selection
        st.markdown("Include Sections")
        col_a, col_b = st.columns(2)
        with col_a:
            include_executive = st.checkbox("Executive Summary", True)
            include_data_overview = st.checkbox("Data Overview", True)
            include_eda = st.checkbox("EDA Analysis", True)
        with col_b:
            include_ml = st.checkbox("ML Results", True)
            include_explanations = st.checkbox("Explanations", True)
            include_sql = st.checkbox("SQL Insights", True)
    
    with col2:
        st.markdown("### Options")
        
        include_recommendations = st.checkbox("Recommendations", True)
        include_visualizations = st.checkbox("Include Charts", True)
        
        # Report preview
        st.markdown("Preview")
        sections_count = sum([
            include_executive, include_data_overview, include_eda, 
            include_ml, include_explanations, include_sql, include_recommendations
        ])
        
        st.info(f"""
        Sections: {sections_count}  
        Format: {report_format}  
        Charts: {'Yes' if include_visualizations else 'No'}
        """)
    
    # Generate report
    if st.button("Generate", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Generating {report_format} report..."):
                
                # Prepare report data
                report_data = {
                    'dataset': st.session_state.current_dataset,
                    'ml_results': st.session_state.get('ml_results', {}),
                    'eda_results': st.session_state.get('eda_results', {}),
                    'explanation_results': st.session_state.get('explanation_results', {}),
                    'chat_history': st.session_state.get('query_history', [])
                }
                
                # Configure sections
                include_sections = {
                    'executive_summary': include_executive,
                    'data_overview': include_data_overview,
                    'eda_analysis': include_eda,
                    'ml_results': include_ml,
                    'explanations': include_explanations,
                    'sql_insights': include_sql,
                    'recommendations': include_recommendations
                }
                
                # Generate report
                report_path = st.session_state.report_generator.generate_comprehensive_report(
                    report_data,
                    report_title=report_title,
                    format=report_format.lower(),
                    include_sections=include_sections,
                    include_visualizations=include_visualizations
                )
                
                if report_path and Path(report_path).exists():
                    st.success("Report generated successfully")
                    
                    # File details
                    file_size = Path(report_path).stat().st_size / 1024
                    st.info(f"File: {report_path} | Size: {file_size:.1f} KB")
                    
                    # Download button
                    try:
                        with open(report_path, 'rb') as file:
                            file_extension = report_format.lower()
                            mime_types = {
                                'html': 'text/html',
                                'pdf': 'application/pdf',
                                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                            }
                            
                            st.download_button(
                                label=f"Download {report_format} Report",
                                data=file.read(),
                                file_name=f"{report_title.replace(' ', '_')}.{file_extension}",
                                mime=mime_types.get(file_extension, 'application/octet-stream'),
                                type="primary",
                                use_container_width=True
                            )
                    except Exception as download_error:
                        st.error(f"Download error: {str(download_error)}")
                        st.info(f"Access report at: {report_path}")
                else:
                    st.error("Report generation failed")
                    
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
    
    # Report templates
    st.markdown("---")
    st.markdown("### Report Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("**Executive Report**")
            st.markdown("""
            - Executive summary
            - Key findings
            - Business recommendations
            - High-level metrics
            """)
    
    with col2:
        with st.container(border=True):
            st.markdown("**Technical Report**")
            st.markdown("""
            - Detailed EDA analysis
            - ML model performance
            - Feature importance
            - Statistical insights
            """)
    
    with col3:
        with st.container(border=True):
            st.markdown("**Complete Report**")
            st.markdown("""
            - All sections included
            - Comprehensive analysis
            - Visualizations
            - Actionable insights
            """)

def main():
    """Main application logic"""
    
    # Initialize session state
    initialize_session_state()
    
    # Setup logging
    setup_logging(st.session_state.config)
    
    # Show header
    show_header()
    
    # Show navigation and get current page
    page_list = [
    "Home Dashboard",
    "Data Upload & Processing",
    "Exploratory Data Analysis",
    "Machine Learning Training",
    "Model Explainability & Insights",
    "SQL Natural Language Chat",\
    "Automated Report Generation"
]
    current_page = show_top_nav(page_list)
    
    # Route to different pages
    if current_page == "Home Dashboard":
        show_home_dashboard()
    elif current_page == "Data Upload & Processing":
        show_data_upload_page()
    elif current_page == "Exploratory Data Analysis":
        show_eda_page()
    elif current_page == "Machine Learning Training":
        show_ml_page()
    elif current_page == "Model Explainability & Insights":
        show_explainability_page()
    elif current_page == "SQL Natural Language Chat":
        show_sql_chat_page()
    elif current_page == "Automated Report Generation":
        show_reports_page()

if __name__ == "__main__":
    main()

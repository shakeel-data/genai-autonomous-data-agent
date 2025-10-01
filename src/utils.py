"""
Utility Functions
Common helper functions used throughout the application
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import streamlit as st
from loguru import logger

def setup_logging(config=None) -> logging.Logger:
    """Setup application logging"""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure loguru
    logger.remove()  # Remove default handler
    
    # Add file handler
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    )
    
    # Add console handler
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )
    
    logger.info("GenAI Autonomous Data Agent - Logging initialized")
    return logger

def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """Load sample datasets for testing"""
    
    sample_datasets = {
        "E-commerce Sales": generate_ecommerce_data(),
        "Healthcare Analytics": generate_healthcare_data(),
        "Financial Market Data": generate_financial_data(),
        "Automotive Performance": generate_automotive_data()
    }
    
    if dataset_name in sample_datasets:
        return sample_datasets[dataset_name]
    else:
        # Return ecommerce as default
        return generate_ecommerce_data()

def generate_ecommerce_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate synthetic e-commerce sales data"""
    
    np.random.seed(42)
    
    # Customer segments
    segments = ['Premium', 'Standard', 'Budget']
    products = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    data = {
        'customer_id': range(1, n_rows + 1),
        'customer_segment': np.random.choice(segments, n_rows, p=[0.2, 0.5, 0.3]),
        'product_category': np.random.choice(products, n_rows),
        'region': np.random.choice(regions, n_rows),
        'order_date': pd.date_range('2023-01-01', periods=n_rows, freq='D')[:n_rows],
        'quantity': np.random.randint(1, 10, n_rows),
        'unit_price': np.random.uniform(10, 500, n_rows).round(2),
        'discount_percent': np.random.uniform(0, 30, n_rows).round(1),
        'customer_age': np.random.randint(18, 70, n_rows),
        'is_repeat_customer': np.random.choice([True, False], n_rows, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived fields
    df['total_amount'] = df['quantity'] * df['unit_price']
    df['discount_amount'] = (df['total_amount'] * df['discount_percent'] / 100).round(2)
    df['final_amount'] = (df['total_amount'] - df['discount_amount']).round(2)
    
    # Add some realistic relationships
    premium_boost = df['customer_segment'] == 'Premium'
    df.loc[premium_boost, 'unit_price'] *= 1.5
    df.loc[premium_boost, 'quantity'] *= 1.2
    
    return df

def generate_healthcare_data(n_rows: int = 800) -> pd.DataFrame:
    """Generate synthetic healthcare analytics data"""
    
    np.random.seed(42)
    
    conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Healthy']
    treatments = ['Treatment_A', 'Treatment_B', 'Treatment_C', 'No_Treatment']
    
    data = {
        'patient_id': range(1, n_rows + 1),
        'age': np.random.randint(20, 90, n_rows),
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'condition': np.random.choice(conditions, n_rows, p=[0.15, 0.2, 0.15, 0.1, 0.4]),
        'treatment': np.random.choice(treatments, n_rows),
        'days_in_hospital': np.random.randint(0, 30, n_rows),
        'treatment_cost': np.random.uniform(1000, 50000, n_rows).round(2),
        'recovery_time': np.random.randint(1, 365, n_rows),
        'readmission': np.random.choice([True, False], n_rows, p=[0.15, 0.85])
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic relationships
    serious_conditions = df['condition'].isin(['Diabetes', 'Heart Disease'])
    df.loc[serious_conditions, 'treatment_cost'] *= 1.5
    df.loc[serious_conditions, 'recovery_time'] *= 1.3
    
    return df

def generate_financial_data(n_rows: int = 1200) -> pd.DataFrame:
    """Generate synthetic financial market data"""
    
    np.random.seed(42)
    
    sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    # Generate time series data
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='H')
    base_price = 100
    
    data = {
        'timestamp': dates,
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], n_rows),
        'sector': np.random.choice(sectors, n_rows),
        'open_price': [],
        'close_price': [],
        'volume': np.random.randint(1000000, 50000000, n_rows),
        'market_cap': np.random.uniform(1e9, 1e12, n_rows)
    }
    
    # Generate realistic price movements
    price_changes = np.random.normal(0, 2, n_rows)  # 2% standard deviation
    prices = [base_price]
    
    for change in price_changes[:-1]:
        new_price = prices[-1] * (1 + change/100)
        prices.append(max(new_price, 1))  # Ensure price stays positive
    
    data['open_price'] = prices
    data['close_price'] = [p * (1 + np.random.normal(0, 0.5)/100) for p in prices]
    
    df = pd.DataFrame(data)
    df['price_change'] = ((df['close_price'] - df['open_price']) / df['open_price'] * 100).round(2)
    
    return df

def generate_automotive_data(n_rows: int = 600) -> pd.DataFrame:
    """Generate synthetic automotive performance data"""
    
    np.random.seed(42)
    
    makes = ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Ford', 'Audi']
    fuel_types = ['Gasoline', 'Diesel', 'Electric', 'Hybrid']
    
    data = {
        'vehicle_id': range(1, n_rows + 1),
        'make': np.random.choice(makes, n_rows),
        'model_year': np.random.randint(2015, 2024, n_rows),
        'fuel_type': np.random.choice(fuel_types, n_rows, p=[0.4, 0.2, 0.2, 0.2]),
        'engine_size': np.random.uniform(1.0, 6.0, n_rows).round(1),
        'horsepower': np.random.randint(100, 500, n_rows),
        'weight': np.random.randint(1200, 2500, n_rows),
        'fuel_efficiency': np.random.uniform(15, 50, n_rows).round(1),
        'maintenance_cost': np.random.uniform(500, 5000, n_rows).round(2),
        'safety_rating': np.random.randint(3, 6, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic relationships
    electric_vehicles = df['fuel_type'] == 'Electric'
    df.loc[electric_vehicles, 'fuel_efficiency'] *= 2  # Electric cars are more efficient
    df.loc[electric_vehicles, 'maintenance_cost'] *= 0.7  # Lower maintenance
    
    luxury_makes = df['make'].isin(['BMW', 'Mercedes', 'Audi'])
    df.loc[luxury_makes, 'maintenance_cost'] *= 1.5  # Higher maintenance for luxury
    
    return df

def format_number(number: Union[int, float]) -> str:
    """Format numbers for display"""
    if isinstance(number, float):
        if abs(number) >= 1e9:
            return f"{number/1e9:.2f}B"
        elif abs(number) >= 1e6:
            return f"{number/1e6:.2f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.2f}K"
        else:
            return f"{number:.2f}"
    else:
        if abs(number) >= 1e9:
            return f"{number/1e9:.2f}B"
        elif abs(number) >= 1e6:
            return f"{number/1e6:.2f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.2f}K"
        else:
            return str(number)

def create_download_link(df: pd.DataFrame, filename: str, file_format: str = 'csv') -> str:
    """Create download link for dataframes"""
    if file_format.lower() == 'csv':
        csv = df.to_csv(index=False)
        return csv
    elif file_format.lower() == 'excel':
        return df.to_excel(index=False)

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate uploaded dataframe"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
    }
    
    # Check for empty dataframe
    if len(df) == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    # Check for too many missing values
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percentage > 50:
        validation_results['warnings'].append(f"High missing values: {missing_percentage:.1f}%")
    
    # Check for duplicate rows
    if df.duplicated().sum() > len(df) * 0.1:
        validation_results['warnings'].append(f"High duplicate rows: {df.duplicated().sum()}")
    
    return validation_results

def safe_execute(func, *args, **kwargs):
    """Safely execute functions with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        return None

# Streamlit helper functions
def show_dataframe_info(df: pd.DataFrame):
    """Display dataframe information in Streamlit"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")

def progress_bar(current: int, total: int, text: str = "Processing"):
    """Display progress bar"""
    progress = current / total
    st.progress(progress, text=f"{text}: {current}/{total}")
